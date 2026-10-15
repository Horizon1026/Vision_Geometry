#include "point_triangulator.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

bool PointTriangulator::Triangulate(const std::vector<Vec3> &p_wc, const std::vector<Quat> &q_wc, const std::vector<Vec3> &bearing_xyz, Vec3 &p_w) {
    RETURN_FALSE_IF(q_wc.size() < 2);
    RETURN_FALSE_IF(q_wc.size() != p_wc.size() || q_wc.size() != bearing_xyz.size());
    switch (options_.kMethod) {
        default:
        case Method::kAnalytic: {
            return TriangulateAnalytic(p_wc, q_wc, bearing_xyz, p_w);
        }
        case Method::kOptimize:
        case Method::kOptimizeHuber:
        case Method::kOptimizeCauchy: {
            return TriangulateIterative(p_wc, q_wc, bearing_xyz, p_w);
        }
    }
    return false;
}

bool PointTriangulator::TriangulateAnalytic(const std::vector<Vec3> &p_wc, const std::vector<Quat> &q_wc, const std::vector<Vec3> &bearing_xyz, Vec3 &p_w) {
    RETURN_FALSE_IF(q_wc.size() < 2);
    RETURN_FALSE_IF(q_wc.size() != p_wc.size() || q_wc.size() != bearing_xyz.size());

    const uint32_t used_camera_num = options_.kMaxUsedCameraView < q_wc.size() ? options_.kMaxUsedCameraView : q_wc.size();
    Eigen::Matrix<float, Eigen::Dynamic, 4> A = Eigen::Matrix<float, Eigen::Dynamic, 4>::Zero(used_camera_num * 3, 4);

    for (uint32_t i = 0; i < used_camera_num; ++i) {
        const Mat3 R_wc = q_wc[i].toRotationMatrix();
        const Vec3 t_wc = p_wc[i];
        const Vec3 b_c = bearing_xyz[i];

        // Construct constraint equation: b_c × (R_cw * (p_w - t_wc)) = 0
        const Mat3 R_cw = R_wc.transpose();
        const Mat3 b_c_skew = Utility::SkewSymmetricMatrix(b_c);
        const Mat3 constraint_mat = b_c_skew * R_cw;
        const Vec3 constraint_vec = b_c_skew * R_cw * t_wc;

        A.block<3, 3>(3 * i, 0) = constraint_mat;
        A.block<3, 1>(3 * i, 3) = -constraint_vec;
    }

    RETURN_FALSE_IF_FALSE(SolveAnalyticConstraint(A, p_w));

    return CheckResultInMultiView(p_wc, q_wc, bearing_xyz, p_w);
}

bool PointTriangulator::TriangulateIterative(const std::vector<Vec3> &p_wc, const std::vector<Quat> &q_wc, const std::vector<Vec3> &bearing_xyz, Vec3 &p_w) {
    RETURN_FALSE_IF(q_wc.size() < 2);
    RETURN_FALSE_IF(q_wc.size() != p_wc.size() || q_wc.size() != bearing_xyz.size());

    uint32_t used_camera_num = options_.kMaxUsedCameraView < q_wc.size() ? options_.kMaxUsedCameraView : q_wc.size();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        Mat3 hessian = Mat3::Zero();
        Vec3 bias = Vec3::Zero();

        for (uint32_t i = 0; i < used_camera_num; ++i) {
            const Vec3 p_c = q_wc[i].inverse() * (p_w - p_wc[i]);
            const float p_c_norm = p_c.norm();
            CONTINUE_IF(p_c_norm < options_.kMinValidDepth || std::isnan(p_c_norm));
            const Vec3 b_c_pred = p_c.normalized();
            const Vec3 b_c_obs = bearing_xyz[i];

            // Compute residual and jacobian.
            const Vec3 residual = b_c_pred - b_c_obs;
            Mat3x3 jacobian_b_pred = Mat3::Zero();
            jacobian_b_pred = (Mat3::Identity() * p_c_norm - p_c * p_c.transpose() / p_c_norm) / (p_c_norm * p_c_norm);
            const Mat3x3 jacobian = jacobian_b_pred * q_wc[i].inverse().toRotationMatrix();

            // Construct incremental equation.
            switch (options_.kMethod) {
                default:
                case Method::kOptimize: {
                    hessian += jacobian.transpose() * jacobian;
                    bias -= jacobian.transpose() * residual;
                    break;
                }
                case Method::kOptimizeHuber: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Huber(options_.kDefaultHuberKernelParameter, r_norm);
                    hessian += jacobian.transpose() * jacobian * kernel;
                    bias -= jacobian.transpose() * residual * kernel;
                    break;
                }
                case Method::kOptimizeCauchy: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Cauchy(options_.kDefaultCauchyKernelParameter, r_norm);
                    hessian += jacobian.transpose() * jacobian * kernel;
                    bias -= jacobian.transpose() * residual * kernel;
                    break;
                }
            }
        }

        // Solve incremental equation and update.
        const Vec3 dx = hessian.ldlt().solve(bias);
        RETURN_FALSE_IF(Eigen::isnan(dx.array()).any());
        p_w += dx;
        BREAK_IF(dx.norm() < options_.kMaxConvergeStep);
    }
    return CheckResultInMultiView(p_wc, q_wc, bearing_xyz, p_w);
}

bool PointTriangulator::CheckResultInMultiView(const std::vector<Vec3> &p_wc, const std::vector<Quat> &q_wc, const std::vector<Vec3> &bearing_xyz,
                                               const Vec3 &p_w) {
    for (uint32_t i = 0; i < q_wc.size(); ++i) {
        const Vec3 p_c = q_wc[i].inverse() * (p_w - p_wc[i]);
        RETURN_FALSE_IF(p_c.squaredNorm() <= kZeroFloat);
        const Vec3 pred_bearing_xyz = p_c.normalized();
        RETURN_FALSE_IF((bearing_xyz[i] - pred_bearing_xyz).norm() > options_.kMaxToleranceReprojectionError);
    }

    return true;
}

float PointTriangulator::GetSineOfParallexAngle(const Vec3 &p_wci, const Quat &q_wci, const Vec3 &p_wcj, const Quat &q_wcj, const Vec3 &bearing_xyz_i,
                                                const Vec3 &bearing_xyz_j) {
    const Quat q_cjci = q_wcj.inverse() * q_wci;
    return (bearing_xyz_j.cross(q_cjci * bearing_xyz_i)).norm();
}

}  // namespace vision_geometry
