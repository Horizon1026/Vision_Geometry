#include "geometry_homography_pnp.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include "cmath"
#include "random"
#include "set"

namespace vision_geometry {

bool HomographyPnpSolver::EstimatePose(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz,
                                       Vec3 &p_cr, Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(all_ref_bearing_xyz.size() != all_cur_bearing_xyz.size());
    RETURN_FALSE_IF(all_ref_bearing_xyz.size() < 5);
    RETURN_FALSE_IF(ref_depth_m < kZeroFloat);
    RETURN_FALSE_IF(!std::isfinite(ref_depth_m) || !p_cr.allFinite() || !homo_plane_slope_xy.allFinite());
    RETURN_FALSE_IF(!q_cr.coeffs().allFinite() || q_cr.norm() < kZeroFloat);
    for (uint32_t i = 0; i < all_ref_bearing_xyz.size(); ++i) {
        RETURN_FALSE_IF(!all_ref_bearing_xyz[i].allFinite() || !all_cur_bearing_xyz[i].allFinite());
        RETURN_FALSE_IF(all_ref_bearing_xyz[i].norm() < kZeroFloat || all_cur_bearing_xyz[i].norm() < kZeroFloat);
    }
    q_cr.normalize();

    switch (options_.kMethod) {
        case Method::kOptimizeRansac:
            return EstimatePoseRansac(all_ref_bearing_xyz, ref_depth_m, all_cur_bearing_xyz, p_cr, q_cr, homo_plane_slope_xy, status);
        default:
        case Method::kOptimize:
        case Method::kOptimizeHuber:
        case Method::kOptimizeCauchy:
            return EstimatePoseUseAll(all_ref_bearing_xyz, ref_depth_m, all_cur_bearing_xyz, p_cr, q_cr, homo_plane_slope_xy, status);
    }

    return true;
}

bool HomographyPnpSolver::ProjectBearing(const Vec3 &ref_bearing_xyz, const float ref_depth_m, const Vec2 &homo_plane_slope_xy, const Vec3 &p_cr,
                                         const Quat &q_cr, Vec3 &cur_bearing_xyz) {
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
                                             Vec3 &p_cr, Quat &q_cr, Vec2 &homo_plane_slope_xy) {
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
            RETURN_FALSE_IF_FALSE(ProjectBearing(ref_bearing_xyz, ref_depth_m, homo_plane_slope_xy, p_cr, q_cr, pred_cur_bearing_xyz));
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


bool HomographyPnpSolver::EstimatePoseUseAll(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz,
                                             Vec3 &p_cr, Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(EstimatePoseUseAll(all_ref_bearing_xyz, ref_depth_m, all_cur_bearing_xyz, p_cr, q_cr, homo_plane_slope_xy));

    if (status.size() != all_ref_bearing_xyz.size()) {
        status.assign(all_ref_bearing_xyz.size(), static_cast<uint8_t>(Result::kUnsolved));
    }

    // Check those features that haven't been solved.
    for (uint32_t i = 0; i < all_ref_bearing_xyz.size(); ++i) {
        CONTINUE_IF(status[i] != static_cast<uint8_t>(Result::kUnsolved));
        const Vec3 &ref_bearing_xyz = all_ref_bearing_xyz[i];
        const Vec3 &cur_bearing_xyz = all_cur_bearing_xyz[i];
        Vec3 pred_cur_bearing_xyz = Vec3::Zero();
        if (ProjectBearing(ref_bearing_xyz, ref_depth_m, homo_plane_slope_xy, p_cr, q_cr, pred_cur_bearing_xyz)) {
            const float residual = (pred_cur_bearing_xyz - cur_bearing_xyz).norm();
            if (residual < options_.kMaxPnpResidual) {
                status[i] = static_cast<uint8_t>(Result::kSolved);
                continue;
            }
        }
        status[i] = static_cast<uint8_t>(Result::kLargeResidual);
    }

    return true;
}

bool HomographyPnpSolver::EstimatePoseRansac(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz,
                                             Vec3 &p_cr, Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(all_ref_bearing_xyz.size() < 5 || all_ref_bearing_xyz.size() != all_cur_bearing_xyz.size());
    RETURN_FALSE_IF(options_.kMaxIteration <= 0);
    RETURN_FALSE_IF(!status.empty() && status.size() != all_ref_bearing_xyz.size());

    // Select only tracked pairs; an empty status means that all pairs are available.
    std::vector<uint32_t> tracked_indices;
    tracked_indices.reserve(all_ref_bearing_xyz.size());
    for (uint32_t i = 0; i < all_ref_bearing_xyz.size(); ++i) {
        CONTINUE_IF(!status.empty() && status[i] != static_cast<uint8_t>(Result::kSolved));
        tracked_indices.emplace_back(i);
    }
    RETURN_FALSE_IF(tracked_indices.size() < 5);

    Vec3 best_p_cr = p_cr;
    Quat best_q_cr = q_cr;
    Vec2 best_slope_xy = homo_plane_slope_xy;
    uint32_t best_score = 0;
    float best_residual = 0.0f;
    std::set<uint32_t> indices;
    std::vector<Vec3> sub_ref_bearings;
    std::vector<Vec3> sub_cur_bearings;
    sub_ref_bearings.reserve(5);
    sub_cur_bearings.reserve(5);
    std::mt19937 rng(std::random_device {}());
    std::uniform_int_distribution<uint32_t> distribution(0, tracked_indices.size() - 1);

    for (int32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Estimate the pose and plane slopes from five distinct bearing pairs.
        indices.clear();
        sub_ref_bearings.clear();
        sub_cur_bearings.clear();
        while (indices.size() < 5) {
            indices.insert(tracked_indices[distribution(rng)]);
        }
        for (const auto i: indices) {
            sub_ref_bearings.emplace_back(all_ref_bearing_xyz[i]);
            sub_cur_bearings.emplace_back(all_cur_bearing_xyz[i].normalized());
        }
        Vec3 candidate_p_cr = p_cr;
        Quat candidate_q_cr = q_cr;
        Vec2 candidate_slope_xy = homo_plane_slope_xy;
        CONTINUE_IF(!EstimatePoseUseAll(sub_ref_bearings, ref_depth_m, sub_cur_bearings, candidate_p_cr, candidate_q_cr, candidate_slope_xy));

        // Score each model on the selected pairs using the unit-sphere distance.
        uint32_t score = 0;
        float total_residual = 0.0f;
        for (const auto i: tracked_indices) {
            Vec3 predicted_bearing = Vec3::Zero();
            CONTINUE_IF(!ProjectBearing(all_ref_bearing_xyz[i], ref_depth_m, candidate_slope_xy, candidate_p_cr, candidate_q_cr, predicted_bearing));
            const float residual = (predicted_bearing - all_cur_bearing_xyz[i].normalized()).norm();
            if (residual < options_.kMaxPnpResidual) {
                ++score;
                total_residual += residual;
            }
        }
        if (score > best_score || (score == best_score && total_residual < best_residual)) {
            best_score = score;
            best_residual = total_residual;
            best_p_cr = candidate_p_cr;
            best_q_cr = candidate_q_cr;
            best_slope_xy = candidate_slope_xy;
        }
        BREAK_IF(static_cast<float>(best_score) / tracked_indices.size() > options_.kMinRansacInlierRatio);
    }
    RETURN_FALSE_IF(best_score < 5);

    // Refine the best model using its consensus set instead of retaining a five-pair fit.
    sub_ref_bearings.clear();
    sub_cur_bearings.clear();
    for (const auto i: tracked_indices) {
        Vec3 predicted_bearing = Vec3::Zero();
        CONTINUE_IF(!ProjectBearing(all_ref_bearing_xyz[i], ref_depth_m, best_slope_xy, best_p_cr, best_q_cr, predicted_bearing));
        CONTINUE_IF((predicted_bearing - all_cur_bearing_xyz[i].normalized()).norm() >= options_.kMaxPnpResidual);
        sub_ref_bearings.emplace_back(all_ref_bearing_xyz[i]);
        sub_cur_bearings.emplace_back(all_cur_bearing_xyz[i].normalized());
    }
    Vec3 refined_p_cr = best_p_cr;
    Quat refined_q_cr = best_q_cr;
    Vec2 refined_slope_xy = best_slope_xy;
    if (EstimatePoseUseAll(sub_ref_bearings, ref_depth_m, sub_cur_bearings, refined_p_cr, refined_q_cr, refined_slope_xy)) {
        uint32_t score = 0;
        float total_residual = 0.0f;
        for (const auto i: tracked_indices) {
            Vec3 predicted_bearing = Vec3::Zero();
            CONTINUE_IF(!ProjectBearing(all_ref_bearing_xyz[i], ref_depth_m, refined_slope_xy, refined_p_cr, refined_q_cr, predicted_bearing));
            const float residual = (predicted_bearing - all_cur_bearing_xyz[i].normalized()).norm();
            if (residual < options_.kMaxPnpResidual) {
                ++score;
                total_residual += residual;
            }
        }
        if (score > best_score || (score == best_score && total_residual <= best_residual)) {
            best_p_cr = refined_p_cr;
            best_q_cr = refined_q_cr;
            best_slope_xy = refined_slope_xy;
        }
    }

    // Publish the best model and update only selected pairs, preserving excluded states.
    p_cr = best_p_cr;
    q_cr = best_q_cr;
    homo_plane_slope_xy = best_slope_xy;
    if (status.empty()) {
        status.assign(all_ref_bearing_xyz.size(), static_cast<uint8_t>(Result::kUnsolved));
    }
    for (const auto i: tracked_indices) {
        status[i] = static_cast<uint8_t>(Result::kLargeResidual);
        Vec3 predicted_bearing = Vec3::Zero();
        CONTINUE_IF(!ProjectBearing(all_ref_bearing_xyz[i], ref_depth_m, homo_plane_slope_xy, p_cr, q_cr, predicted_bearing));
        if ((predicted_bearing - all_cur_bearing_xyz[i].normalized()).norm() < options_.kMaxPnpResidual) {
            status[i] = static_cast<uint8_t>(Result::kSolved);
        }
    }
    return true;
}

}  // namespace vision_geometry
