#include "line_triangulator.h"
#include "slam_basic_math.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"
#include "plane.h"

namespace VISION_GEOMETRY {

bool LineTriangulator::Triangulate(const std::vector<Quat> &all_q_wc,
                                   const std::vector<Vec3> &all_p_wc,
                                   const std::vector<LineSegment2D> &lines_in_norm_plane,
                                   LinePlucker3D &plucker_in_w) {
    RETURN_FALSE_IF(all_q_wc.size() < 2);
    RETURN_FALSE_IF(all_q_wc.size() != all_p_wc.size() || all_q_wc.size() != lines_in_norm_plane.size());

    switch (options_.kMethod) {
        default:
        case TriangulationMethod::kAnalytic: {
            return TriangulateAnalytic(all_q_wc, all_p_wc, lines_in_norm_plane, plucker_in_w);
        }
        case TriangulationMethod::kIterative: {
            return TriangulateIterative(all_q_wc, all_p_wc, lines_in_norm_plane, plucker_in_w);
        }
    }
    return false;
}

bool LineTriangulator::TriangulateAnalytic(const std::vector<Quat> &all_q_wc,
                                           const std::vector<Vec3> &all_p_wc,
                                           const std::vector<LineSegment2D> &lines_in_norm_plane,
                                           LinePlucker3D &plucker_in_w) {
    // Generate plane from first camera view.
    const Quat &q_wc1 = all_q_wc[0];
    const Vec3 &p_wc1 = all_p_wc[0];
    const Vec3 point1_from_c1_in_w = q_wc1 * lines_in_norm_plane[0].start_point_homogeneous() + p_wc1;
    const Vec3 point2_from_c1_in_w = q_wc1 * lines_in_norm_plane[0].end_point_homogeneous() + p_wc1;
    Plane3D plane_of_c1;
    RETURN_FALSE_IF_FALSE(plane_of_c1.FitPlaneModel(p_wc1, point1_from_c1_in_w, point2_from_c1_in_w));

    // Generate plane from second camera view.
    const Quat &q_wc2 = all_q_wc[1];
    const Vec3 &p_wc2 = all_p_wc[1];
    const Vec3 point1_from_c2_in_w = q_wc2 * lines_in_norm_plane[1].start_point_homogeneous() + p_wc2;
    const Vec3 point2_from_c2_in_w = q_wc2 * lines_in_norm_plane[1].end_point_homogeneous() + p_wc2;
    Plane3D plane_of_c2;
    RETURN_FALSE_IF_FALSE(plane_of_c2.FitPlaneModel(p_wc2, point1_from_c2_in_w, point2_from_c2_in_w));

    // Estimate line in plucker.
    const Mat4 dual_plucker_matrix = plane_of_c1.param() * plane_of_c2.param().transpose() -
        plane_of_c2.param() * plane_of_c1.param().transpose();
    plucker_in_w = LinePlucker3D(dual_plucker_matrix);
    return true;
}

bool LineTriangulator::TriangulateIterative(const std::vector<Quat> &all_q_wc,
                                            const std::vector<Vec3> &all_p_wc,
                                            const std::vector<LineSegment2D> &lines_in_norm_plane,
                                            LinePlucker3D &plucker_in_w) {
    // Try to get initialized parameters.
    if (!plucker_in_w.SelfCheck()) {
        TriangulateAnalytic(all_q_wc, all_p_wc, lines_in_norm_plane, plucker_in_w);
    }

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        LineOrthonormal3D orthonormal_in_w(plucker_in_w);
        const Mat3 matrix_U = orthonormal_in_w.matrix_U();
        const Mat2 matrix_W = orthonormal_in_w.matrix_W();
        const float w1 = matrix_W(0, 0);
        const float w2 = matrix_W(1, 0);

        Mat4 hessian = Mat4::Zero();
        Vec4 bias = Vec4::Zero();
        for (uint32_t i = 0; i < lines_in_norm_plane.size(); ++i) {
            BREAK_IF(i >= options_.kMaxUsedCameraView);
            const LinePlucker3D plucker_in_c = plucker_in_w.TransformTo(all_q_wc[i], all_p_wc[i]);
            const Vec3 l = plucker_in_c.ProjectToNormalPlane();
            const float l_squared_norm = l.head<2>().squaredNorm();
            const float l_norm = std::sqrt(l_squared_norm);
            const float l_squared_3_2 = l_squared_norm * l_norm;

            const Mat3 R_wc(all_q_wc[i]);
            const Vec3 p_wc = all_p_wc[i];
            const auto &line_in_norm_plane = lines_in_norm_plane[i];
            const Vec3 s_point = line_in_norm_plane.start_point_homogeneous();
            const Vec3 e_point = line_in_norm_plane.end_point_homogeneous();

            // Compute residual. Define it by distance from point to line.
            const Vec2 residual = Vec2(s_point.dot(l) / l_norm,
                                       e_point.dot(l) / l_norm);

            // Compute jacobian of d_residual to d_line_in_c.
            Mat2x3 jacobian_residual_line_in_c = Mat2x3::Zero();
            jacobian_residual_line_in_c << - l[0] * s_point.dot(l) / l_squared_3_2 + s_point.x() / l_norm,
                                           - l[1] * s_point.dot(l) / l_squared_3_2 + s_point.y() / l_norm,
                                           1.0f / l_norm,
                                           - l[0] * e_point.dot(l) / l_squared_3_2 + e_point.x() / l_norm,
                                           - l[1] * e_point.dot(l) / l_squared_3_2 + e_point.y() / l_norm,
                                           1.0f / l_norm;
            // Compute jacobian of d_line_in_c to d_plucker_in_c.
            Mat3x6 jacobian_line_to_plucker = Mat3x6::Zero();
            jacobian_line_to_plucker.block<3, 3>(0, 0).setIdentity();
            // Compute jacobian of d_plucker_in_c to d_plucker_in_w.
            Mat6 jacobian_plucker_c_to_w = Mat6::Zero();
            jacobian_plucker_c_to_w.block<3, 3>(0, 0) = R_wc;
            jacobian_plucker_c_to_w.block<3, 3>(3, 3) = R_wc;
            jacobian_plucker_c_to_w.block<3, 3>(0, 3) = Utility::SkewSymmetricMatrix(p_wc) * R_wc;
            // Compute jacobian of d_plucker_in_w to d_orthonormal_in_w.
            Mat6x4 jacobian_plucker_to_orthonormal = Mat6x4::Zero();
            jacobian_plucker_to_orthonormal.block<3, 1>(0, 1) = - w1 * matrix_U.col(2);
            jacobian_plucker_to_orthonormal.block<3, 1>(0, 2) = w1 * matrix_U.col(1);
            jacobian_plucker_to_orthonormal.block<3, 1>(0, 3) = - w2 * matrix_U.col(0);
            jacobian_plucker_to_orthonormal.block<3, 1>(3, 0) = w2 * matrix_U.col(2);
            jacobian_plucker_to_orthonormal.block<3, 1>(3, 2) = - w2 * matrix_U.col(0);
            jacobian_plucker_to_orthonormal.block<3, 1>(3, 3) = w1 * matrix_U.col(1);
            // Compute full jacobian.
            const Mat2x4 jacobian = jacobian_residual_line_in_c *
                                    jacobian_line_to_plucker *
                                    jacobian_plucker_c_to_w *
                                    jacobian_plucker_to_orthonormal;

            // Generate incremental function to solve dx.
            hessian += jacobian.transpose() * jacobian;
            bias -= jacobian.transpose() * residual;
        }

        // Solve dx and update parameter of line.
        const Vec4 dx = hessian.ldlt().solve(bias);
        RETURN_FALSE_IF(Eigen::isnan(dx.array()).any());
        orthonormal_in_w.param() += dx;
        plucker_in_w = LinePlucker3D(orthonormal_in_w);
        BREAK_IF(dx.norm() < options_.kMaxConvergeStep);
    }

    return true;
}

}
