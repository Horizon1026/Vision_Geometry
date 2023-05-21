#include "geometry_pnp.h"
#include "math_kinematics.h"
#include "slam_operations.h"

#include <set>

namespace VISION_GEOMETRY {

bool PnpSolver::EstimatePose(const std::vector<Vec3> &p_w,
                             const std::vector<Vec2> &norm_uv,
                             Quat &q_wc,
                             Vec3 &p_wc,
                             std::vector<uint8_t> &status) {
    switch (options_.kMethod) {
        case PnpMethod::PNP_RANSAC: {
            return EstimatePoseRansac(p_w, norm_uv, q_wc, p_wc, status);
        }

        case PnpMethod::PNP_ALL:
        case PnpMethod::PNP_HUBER:
        case PnpMethod::PNP_CAUCHY: {
            return EstimatePoseUseAll(p_w, norm_uv, q_wc, p_wc, status);
        }

        default: {
            return false;
        }
    }
}

bool PnpSolver::EstimatePoseUseAll(const std::vector<Vec3> &p_w,
                                   const std::vector<Vec2> &norm_uv,
                                   Quat &q_wc,
                                   Vec3 &p_wc,
                                   std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(EstimatePoseUseAll(p_w, norm_uv, q_wc, p_wc));

    if (status.size() != p_w.size()) {
        status.resize(p_w.size(), static_cast<uint8_t>(PnpResult::kUnsolved));
    }

    // Check those features that haven't been solved.
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(PnpResult::kUnsolved)) {
            const Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            if (p_c(2) > kZero) {
                const float residual = (norm_uv[i] - p_c.head<2>() / p_c(2)).norm();
                if (residual < options_.kMaxPnpResidual) {
                    status[i] = static_cast<uint8_t>(PnpResult::kSolved);
                } else {
                    status[i] = static_cast<uint8_t>(PnpResult::kLargeResidual);
                }
            }
        }
    }

    return true;
}

bool PnpSolver::EstimatePoseUseAll(const std::vector<Vec3> &p_w,
                                   const std::vector<Vec2> &norm_uv,
                                   Quat &q_wc,
                                   Vec3 &p_wc) {
    if (p_w.size() != norm_uv.size() || p_w.empty()) {
        return false;
    }

    Mat6 H;
    Vec6 b;
    Mat2x6 jacobian;

    uint32_t max_points_used_num = options_.kMaxSolvePointsNumber < p_w.size() ? options_.kMaxSolvePointsNumber : p_w.size();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        H.setZero();
        b.setZero();

        for (uint32_t i = 0; i < max_points_used_num; ++i) {
            Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            if (p_c.z() < options_.kMinValidDepth || std::isnan(p_c.z())) {
                continue;
            }

            const float inv_depth = 1.0f / p_c.z();
            const float inv_depth2 = inv_depth * inv_depth;

            Vec2 residual = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_uv[i];

            Mat2x3 jacobian_2d_3d;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth2,
                              0, inv_depth, - p_c(1) * inv_depth2;
            jacobian.block<2, 3>(0, 0) = jacobian_2d_3d * (- q_wc.inverse().matrix());
            jacobian.block<2, 3>(0, 3) = jacobian_2d_3d * Utility::SkewSymmetricMatrix(p_c);

            switch (options_.kMethod) {
                default:
                case PnpMethod::PNP_ALL: {
                    H += jacobian.transpose() * jacobian;
                    b += - jacobian.transpose() * residual;
                    break;
                }
                case PnpMethod::PNP_HUBER: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Huber(1.0f, r_norm);
                    H += jacobian.transpose() * jacobian * kernel;
                    b += - jacobian.transpose() * residual * kernel;
                    break;
                }
                case PnpMethod::PNP_CAUCHY: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Cauchy(1.0f, r_norm);
                    H += jacobian.transpose() * jacobian * kernel;
                    b += - jacobian.transpose() * residual * kernel;
                    break;
                }
            }
        }

        Vec6 dx = H.ldlt().solve(b);

        float norm_dx = dx.squaredNorm();
        if (std::isnan(norm_dx) == true) {
            return false;
        }
        norm_dx = std::sqrt(norm_dx);

        p_wc += dx.head<3>();
        q_wc = (q_wc * Utility::DeltaQ(dx.tail<3>())).normalized();

        if (norm_dx < options_.kMaxConvergeStep) {
            break;
        }
    }

    return true;
}

bool PnpSolver::EstimatePoseRansac(const std::vector<Vec3> &p_w,
                                   const std::vector<Vec2> &norm_uv,
                                   Quat &q_wc,
                                   Vec3 &p_wc,
                                   std::vector<uint8_t> &status) {

    if (p_w.size() != norm_uv.size() || p_w.empty()) {
        return false;
    }

    Quat best_q_wc = q_wc;
    Vec3 best_p_wc = p_wc;
    uint32_t best_score = 0;
    uint32_t score = 0;

    std::set<uint32_t> indice;
    std::vector<Vec3> subPts3d;
    std::vector<Vec2> subPts2d;
    subPts3d.reserve(3);
    subPts2d.reserve(3);

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Select 3 points randomly.
        indice.clear();
        subPts3d.clear();
        subPts2d.clear();

        while (indice.size() < 3) {
            indice.insert(std::rand() % p_w.size());
        }

        for (auto it = indice.begin(); it != indice.end(); ++it) {
            subPts3d.emplace_back(p_w[*it]);
            subPts2d.emplace_back(norm_uv[*it]);
        }

        // Compute pnp model with 3 points.
        q_wc = best_q_wc;
        p_wc = best_p_wc;
        EstimatePoseUseAll(subPts3d, subPts2d, q_wc, p_wc);

        // Apply pnp model on all points, statis inliers.
        score = 0;
        for (uint32_t i = 0; i < p_w.size(); ++i) {
            Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            Vec2 r = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_uv[i];
            if (r.squaredNorm() < options_.kMaxConvergeResidual) {
                ++score;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_q_wc = q_wc;
            best_p_wc = p_wc;
        }

        if (float(score) / float(p_w.size()) > options_.kMinRansacInlierRatio) {
            break;
        }
    }

    CheckPnpStatus(p_w, norm_uv, q_wc, p_wc, status);

    return true;
}

void PnpSolver::CheckPnpStatus(const std::vector<Vec3> &p_w,
                               const std::vector<Vec2> &norm_uv,
                               Quat &q_wc,
                               Vec3 &p_wc,
                               std::vector<uint8_t> &status) {
    if (status.size() != norm_uv.size()) {
        status.resize(norm_uv.size(), static_cast<uint8_t>(PnpResult::kUnsolved));
    }

    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(PnpResult::kUnsolved) || status[i] == static_cast<uint8_t>(PnpResult::kSolved)) {
            Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            Vec2 r = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_uv[i];
            if (r.squaredNorm() >= options_.kMaxConvergeResidual) {
                status[i] = static_cast<uint8_t>(PnpResult::kLargeResidual);
            } else {
                status[i] = static_cast<uint8_t>(PnpResult::kSolved);
            }
        }
    }
}

}
