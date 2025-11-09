#include "geometry_pnp.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include <set>

namespace VISION_GEOMETRY {

bool PnpSolver::EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                             Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status) {
    switch (options_.kMethod) {
        case Method::kOptimizeRansac:
            return EstimatePoseRansac(p_w, q_ic, p_ic, norm_xy, q_wi, p_wi, status);
        case Method::kOptimize:
        case Method::kOptimizeHuber:
        case Method::kOptimizeCauchy:
            return EstimatePoseUseAll(p_w, q_ic, p_ic, norm_xy, q_wi, p_wi, status);
        case Method::kDirectLinearTransform:
        default:
            return false;
    }
}

bool PnpSolver::EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                                   Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(EstimatePoseUseAll(p_w, q_ic, p_ic, norm_xy, q_wi, p_wi));

    if (status.size() != p_w.size()) {
        status.resize(p_w.size(), static_cast<uint8_t>(Result::kUnsolved));
    }

    // Check those features that haven't been solved.
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(Result::kUnsolved)) {
            const Vec3 p_wc = q_wi * p_ic[i] + p_wi;
            const Quat q_wc = q_wi * q_ic[i];
            const Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            if (p_c(2) > kZeroFloat) {
                const float residual = (norm_xy[i] - p_c.head<2>() / p_c(2)).norm();
                if (residual < options_.kMaxPnpResidual) {
                    status[i] = static_cast<uint8_t>(Result::kSolved);
                } else {
                    status[i] = static_cast<uint8_t>(Result::kLargeResidual);
                }
            }
        }
    }

    return true;
}

bool PnpSolver::EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                                   Quat &q_wi, Vec3 &p_wi) {
    RETURN_FALSE_IF(p_w.size() != norm_xy.size() || p_w.empty());

    Mat6 H = Mat6::Zero();
    Vec6 b = Vec6::Zero();
    ;
    Mat2x6 jacobian = Mat2x6::Zero();
    ;

    uint32_t max_points_used_num = options_.kMaxSolvePointsNumber < p_w.size() ? options_.kMaxSolvePointsNumber : p_w.size();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        H.setZero();
        b.setZero();

        for (uint32_t i = 0; i < max_points_used_num; ++i) {
            const Vec3 p_wc = q_wi * p_ic[i] + p_wi;
            const Quat q_wc = q_wi * q_ic[i];
            const Vec3 p_i = q_wi.inverse() * (p_w[i] - p_wi);
            const Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            CONTINUE_IF(p_c.z() < options_.kMinValidDepth || std::isnan(p_c.z()));
            const float inv_depth = 1.0f / p_c.z();
            const float inv_depth2 = inv_depth * inv_depth;
            const Vec2 residual = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_xy[i];

            Mat2x3 jacobian_2d_3d = Mat2x3::Zero();
            jacobian_2d_3d << inv_depth, 0, -p_c(0) * inv_depth2, 0, inv_depth, -p_c(1) * inv_depth2;
            jacobian.block<2, 3>(0, 0) = jacobian_2d_3d * (-q_wc.inverse().matrix());
            jacobian.block<2, 3>(0, 3) = jacobian_2d_3d * q_ic[i].inverse().matrix() * Utility::SkewSymmetricMatrix(p_i);

            switch (options_.kMethod) {
                default:
                case Method::kOptimize: {
                    H += jacobian.transpose() * jacobian;
                    b += -jacobian.transpose() * residual;
                    break;
                }
                case Method::kOptimizeHuber: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Huber(options_.kDefaultHuberKernelParameter, r_norm);
                    H += jacobian.transpose() * jacobian * kernel;
                    b += -jacobian.transpose() * residual * kernel;
                    break;
                }
                case Method::kOptimizeCauchy: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Cauchy(options_.kDefaultCauchyKernelParameter, r_norm);
                    H += jacobian.transpose() * jacobian * kernel;
                    b += -jacobian.transpose() * residual * kernel;
                    break;
                }
            }
        }

        const Vec6 dx = H.ldlt().solve(b);

        float norm_dx = dx.squaredNorm();
        RETURN_FALSE_IF(std::isnan(norm_dx) == true);
        norm_dx = std::sqrt(norm_dx);

        p_wi += dx.head<3>();
        q_wi = (q_wi * Utility::DeltaQ(dx.tail<3>())).normalized();

        BREAK_IF(norm_dx < options_.kMaxConvergeStep);
    }

    return true;
}

bool PnpSolver::EstimatePoseRansac(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                                   Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(p_w.size() != norm_xy.size() || p_w.empty());

    Quat best_q_wi = q_wi;
    Vec3 best_p_wi = p_wi;
    uint32_t best_score = 0;
    uint32_t score = 0;

    std::set<uint32_t> indice;
    std::vector<Vec3> sub_pts3d;
    std::vector<Quat> sub_q_ic;
    std::vector<Vec3> sub_p_ic;
    std::vector<Vec2> sub_pts2d;
    sub_pts3d.reserve(3);
    sub_q_ic.reserve(3);
    sub_p_ic.reserve(3);
    sub_pts2d.reserve(3);

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Select 3 points randomly.
        indice.clear();
        sub_pts3d.clear();
        sub_q_ic.clear();
        sub_p_ic.clear();
        sub_pts2d.clear();

        while (indice.size() < 3) {
            indice.insert(std::rand() % p_w.size());
        }

        for (auto it = indice.begin(); it != indice.end(); ++it) {
            sub_pts3d.emplace_back(p_w[*it]);
            sub_q_ic.emplace_back(q_ic[*it]);
            sub_p_ic.emplace_back(p_ic[*it]);
            sub_pts2d.emplace_back(norm_xy[*it]);
        }

        // Compute pnp model with 3 points.
        q_wi = best_q_wi;
        p_wi = best_p_wi;
        EstimatePoseUseAll(sub_pts3d, sub_q_ic, sub_p_ic, sub_pts2d, q_wi, p_wi);

        // Apply pnp model on all points, statis inliers.
        score = 0;
        for (uint32_t i = 0; i < p_w.size(); ++i) {
            const Vec3 p_wc = q_wi * p_ic[i] + p_wi;
            const Quat q_wc = q_wi * q_ic[i];
            const Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            const Vec2 r = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_xy[i];
            if (r.squaredNorm() < options_.kMaxConvergeResidual) {
                ++score;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_q_wi = q_wi;
            best_p_wi = p_wi;
        }

        BREAK_IF(float(score) / float(p_w.size()) > options_.kMinRansacInlierRatio);
    }

    CheckPnpStatus(p_w, q_ic, p_ic, norm_xy, q_wi, p_wi, status);
    return true;
}

void PnpSolver::CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                               Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status) {
    if (status.size() != norm_xy.size()) {
        status.resize(norm_xy.size(), static_cast<uint8_t>(Result::kUnsolved));
    }

    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(Result::kUnsolved) || status[i] == static_cast<uint8_t>(Result::kSolved)) {
            const Vec3 p_wc = q_wi * p_ic[i] + p_wi;
            const Quat q_wc = q_wi * q_ic[i];
            const Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            const Vec2 r = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_xy[i];
            if (r.squaredNorm() >= options_.kMaxConvergeResidual) {
                status[i] = static_cast<uint8_t>(Result::kLargeResidual);
            } else {
                status[i] = static_cast<uint8_t>(Result::kSolved);
            }
        }
    }
}

}  // namespace VISION_GEOMETRY
