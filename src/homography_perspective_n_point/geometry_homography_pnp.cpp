#include "geometry_homography_pnp.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include "cmath"

namespace vision_geometry {

bool HomographyPnpSolver::EstimatePose(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz,
                                       Vec3 &p_cr, Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(all_ref_bearing_xyz.size() != all_cur_bearing_xyz.size());
    RETURN_FALSE_IF(all_ref_bearing_xyz.size() < 5);
    RETURN_FALSE_IF(ref_depth_m < kZeroFloat);
    RETURN_FALSE_IF(!std::isfinite(ref_depth_m) || !p_cr.allFinite() || !homo_plane_slope_xy.allFinite());
    RETURN_FALSE_IF(!q_cr.coeffs().allFinite() || q_cr.norm() < kZeroFloat);
    for (size_t i = 0; i < all_ref_bearing_xyz.size(); ++i) {
        RETURN_FALSE_IF(!all_ref_bearing_xyz[i].allFinite() || !all_cur_bearing_xyz[i].allFinite());
        RETURN_FALSE_IF(all_ref_bearing_xyz[i].norm() < kZeroFloat || all_cur_bearing_xyz[i].norm() < kZeroFloat);
    }
    q_cr.normalize();

    switch (options_.kMethod) {
        default:
        case Method::kOptimize:
        case Method::kOptimizeHuber:
        case Method::kOptimizeCauchy:
            return EstimatePoseUseAll(all_ref_bearing_xyz, ref_depth_m, all_cur_bearing_xyz, p_cr, q_cr, homo_plane_slope_xy, status);
    }

    return true;
}

bool HomographyPnpSolver::Project(const Vec3 &ref_bearing_xyz, const float ref_depth_m, const Vec2 &homo_plane_slope_xy, const Vec3 &p_cr, const Quat &q_cr,
                                  Vec3 &cur_bearing_xyz) {
    // Plane model: aX + bY + Z = d
    const float &a = homo_plane_slope_xy.x();
    const float &b = homo_plane_slope_xy.y();
    const float &d = ref_depth_m;

    const float plane_denominator = a * ref_bearing_xyz.x() + b * ref_bearing_xyz.y() + ref_bearing_xyz.z();
    RETURN_FALSE_IF(!std::isfinite(plane_denominator) || plane_denominator <= kZeroFloat);
    const float depth_p_r = d / plane_denominator;
    const Vec3 p_r = ref_bearing_xyz * depth_p_r;
    const Vec3 p_c = q_cr * p_r + p_cr;
    RETURN_FALSE_IF(!p_c.allFinite() || p_c.norm() <= kZeroFloat);
    cur_bearing_xyz = p_c.normalized();
    return true;
}

bool HomographyPnpSolver::EstimatePoseUseAll(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz,
                                             Vec3 &p_cr, Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(all_ref_bearing_xyz.size() < 5);
    RETURN_FALSE_IF(ref_depth_m < kZeroFloat);
    RETURN_FALSE_IF(options_.kMaxIteration <= 0);

    for (int32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Parameters total 8-dof: p_cr, q_cr(theta_cr), plane_slope_x, plane_slope_y
        Mat8 hessian = Mat8::Zero();
        Vec8 bias = Vec8::Zero();

        for (uint32_t i = 0; i < all_ref_bearing_xyz.size(); ++i) {
            const Vec3 &ref_bearing_xyz = all_ref_bearing_xyz[i];
            const Vec3 &cur_bearing_xyz = all_cur_bearing_xyz[i];
            Vec3 pred_cur_bearing_xyz = Vec3::Zero();
            RETURN_FALSE_IF_FALSE(Project(ref_bearing_xyz, ref_depth_m, homo_plane_slope_xy, p_cr, q_cr, pred_cur_bearing_xyz));
            const Vec3 residual = pred_cur_bearing_xyz - cur_bearing_xyz;

            // Compute jacobian.
            const float plane_denominator = homo_plane_slope_xy.x() * ref_bearing_xyz.x() + homo_plane_slope_xy.y() * ref_bearing_xyz.y() + ref_bearing_xyz.z();
            const float depth_p_r = ref_depth_m / plane_denominator;
            const Vec3 p_r = ref_bearing_xyz * depth_p_r;
            const Mat3 R_cr = q_cr.matrix();
            const Vec3 rotated_p_r = R_cr * p_r;
            const Vec3 p_c = rotated_p_r + p_cr;
            const Mat3 jacobian_bearing_p_c = (Mat3::Identity() - pred_cur_bearing_xyz * pred_cur_bearing_xyz.transpose()) / p_c.norm();

            Mat3x8 jacobian = Mat3x8::Zero();
            jacobian.block<3, 3>(0, 0) = jacobian_bearing_p_c;
            jacobian.block<3, 3>(0, 3) = -jacobian_bearing_p_c * R_cr * Utility::SkewSymmetricMatrix(p_r);
            // J_normalize * (R_cr * p_r) = -J_normalize * p_cr.
            // This form makes the unobservable slope columns exactly zero at p_cr = 0.
            jacobian.block<3, 2>(0, 6) = jacobian_bearing_p_c * p_cr * ref_bearing_xyz.head<2>().transpose() / plane_denominator;

            // Construct incremental function.
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
                    bias += -jacobian.transpose() * residual * kernel;
                    break;
                }
                case Method::kOptimizeCauchy: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Cauchy(options_.kDefaultCauchyKernelParameter, r_norm);
                    hessian += jacobian.transpose() * jacobian * kernel;
                    bias += -jacobian.transpose() * residual * kernel;
                    break;
                }
            }
        }

        // Solve incremental function using SVD to handle a rank-deficient Hessian.
        const Vec8 dx = hessian.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(bias);
        RETURN_FALSE_IF(!dx.allFinite());

        // Update parameters.
        p_cr += dx.segment<3>(0);
        q_cr = (q_cr * Utility::Exponent(dx.segment<3>(3))).normalized();
        homo_plane_slope_xy += dx.segment<2>(6);

        if (dx.norm() < options_.kMaxConvergeStep) {
            return true;
        }
    }
    return false;
}

float HomographyPnpSolver::Huber(float param, float x) {
    float huber = 1.0f;
    if (x > param) {
        huber = param / x;
    }
    return huber;
}

float HomographyPnpSolver::Cauchy(float param, float x) {
    const float param2 = param * param;
    return param2 / (param2 + x * x);
}

}  // namespace vision_geometry
