#include "point_triangulator.h"
#include "slam_basic_math.h"
#include "slam_operations.h"

namespace VISION_GEOMETRY {

bool PointTriangulator::Triangulate(const std::vector<Quat> &q_wc,
                                    const std::vector<Vec3> &p_wc,
                                    const std::vector<Vec2> &norm_uv,
                                    Vec3 &p_w) {
    RETURN_FALSE_IF(q_wc.size() < 2);
    RETURN_FALSE_IF(q_wc.size() != p_wc.size() || q_wc.size() != norm_uv.size());
    switch (options_.kMethod) {
        default:
        case TriangulationMethod::kAnalytic: {
            return TriangulateAnalytic(q_wc, p_wc, norm_uv, p_w);
        }
        case TriangulationMethod::kIterative: {
            return TriangulateIterative(q_wc, p_wc, norm_uv, p_w);
        }
    }
    return false;
}

bool PointTriangulator::TriangulateAnalytic(const std::vector<Quat> &q_wc,
                                            const std::vector<Vec3> &p_wc,
                                            const std::vector<Vec2> &norm_uv,
                                            Vec3 &p_w) {
    const uint32_t used_camera_num = options_.kMaxUsedCameraView < q_wc.size() ? options_.kMaxUsedCameraView : q_wc.size();
    Eigen::Matrix<float, Eigen::Dynamic, 4> A = Eigen::Matrix<float, Eigen::Dynamic, 4>::Zero(used_camera_num * 2, 4);
    for (uint32_t i = 0; i < used_camera_num; ++i) {
        auto pose = Utility::TransformMatrix<float>(q_wc[i].inverse(), - (q_wc[i].inverse() * p_wc[i]));
        A.row(2 * i) = norm_uv[i][0] * pose.row(2) - pose.row(0);
        A.row(2 * i + 1) = norm_uv[i][1] * pose.row(2) - pose.row(1);
    }

    const Vec4 x = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    RETURN_FALSE_IF(std::fabs(x(3)) < kZero);
    p_w = x.head<3>() / x(3);

    return CheckDepthInMultiView(q_wc, p_wc, p_w);
}

bool PointTriangulator::TriangulateIterative(const std::vector<Quat> &q_wc,
                                             const std::vector<Vec3> &p_wc,
                                             const std::vector<Vec2> &norm_uv,
                                             Vec3 &p_w) {
    uint32_t used_camera_num = options_.kMaxUsedCameraView < q_wc.size() ? options_.kMaxUsedCameraView : q_wc.size();

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        Mat3 hessian = Mat3::Zero();
        Vec3 bias = Vec3::Zero();

        for (uint32_t i = 0; i < used_camera_num; ++i) {
            const Vec3 p_c = q_wc[i].inverse() * (p_w - p_wc[i]);
            CONTINUE_IF(p_c.z() < options_.kMinValidDepth || std::isnan(p_c.z()));
            const float inv_depth = 1.0f / p_c.z();
            const float inv_depth2 = inv_depth * inv_depth;

            // Compute residual and jacobian.
            const Vec2 residual = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_uv[i];
            Mat2x3 jacobian_2d_3d;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth2,
                              0, inv_depth, - p_c(1) * inv_depth2;
            const Mat2x3 jacobian = jacobian_2d_3d * q_wc[i].inverse().matrix();

            hessian += jacobian.transpose() * jacobian;
            bias -= jacobian.transpose() * residual;
        }

        const Vec3 dx = hessian.ldlt().solve(bias);
        RETURN_FALSE_IF(Eigen::isnan(dx.array()).any());
        p_w += dx;
        BREAK_IF(dx.norm() < options_.kMaxConvergeStep);
    }

    return CheckDepthInMultiView(q_wc, p_wc, p_w);
}

bool PointTriangulator::CheckDepthInMultiView(const std::vector<Quat> &q_wc,
                                              const std::vector<Vec3> &p_wc,
                                              const Vec3 &p_w) {
    for (uint32_t i = 0; i < q_wc.size(); ++i) {
        const Vec3 p_c = q_wc[i].inverse() * (p_w - p_wc[i]);
        RETURN_FALSE_IF(p_c.z() < 0);
    }

    return true;
}

float PointTriangulator::GetParallexAngle(const Quat &q_wci, const Vec3 &p_wci,
                                          const Quat &q_wcj, const Vec3 &p_wcj,
                                          const Vec2 &norm_xy_i, const Vec2 &norm_xy_j) {
    const Vec3 norm_xyz_i = Vec3(norm_xy_i.x(), norm_xy_i.y(), 1.0f);
    const Vec3 norm_xyz_j = Vec3(norm_xy_j.x(), norm_xy_j.y(), 1.0f);
    const Quat q_cjci = q_wcj.inverse() * q_wci;
    return (norm_xyz_j.cross(q_cjci * norm_xyz_i)).norm();
}

}