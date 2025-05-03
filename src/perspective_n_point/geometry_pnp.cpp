#include "geometry_pnp.h"
#include "slam_basic_math.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"

#include <set>

namespace VISION_GEOMETRY {

bool PnpSolver::EstimatePose(const std::vector<Vec3> &p_w,
                             const std::vector<Vec2> &norm_xy,
                             Quat &q_wc,
                             Vec3 &p_wc,
                             std::vector<uint8_t> &status) {
    switch (options_.kMethod) {
        case Method::kOptimizeRansac: {
            return EstimatePoseRansac(p_w, norm_xy, q_wc, p_wc, status);
        }

        case Method::kOptimize:
        case Method::kOptimizeHuber:
        case Method::kOptimizeCauchy: {
            return EstimatePoseUseAll(p_w, norm_xy, q_wc, p_wc, status);
        }

        case Method::kDirectLinearTransform: {
            return EstimatePoseDlt(p_w, norm_xy, q_wc, p_wc, status);
        }

        default: {
            return false;
        }
    }
}

bool PnpSolver::EstimatePoseUseAll(const std::vector<Vec3> &p_w,
                                   const std::vector<Vec2> &norm_xy,
                                   Quat &q_wc,
                                   Vec3 &p_wc,
                                   std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(EstimatePoseUseAll(p_w, norm_xy, q_wc, p_wc));

    if (status.size() != p_w.size()) {
        status.resize(p_w.size(), static_cast<uint8_t>(Result::kUnsolved));
    }

    // Check those features that haven't been solved.
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(Result::kUnsolved)) {
            const Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            if (p_c(2) > kZerofloat) {
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

bool PnpSolver::EstimatePoseUseAll(const std::vector<Vec3> &p_w,
                                   const std::vector<Vec2> &norm_xy,
                                   Quat &q_wc,
                                   Vec3 &p_wc) {
    RETURN_FALSE_IF(p_w.size() != norm_xy.size() || p_w.empty());

    Mat6 H = Mat6::Zero();
    Vec6 b = Vec6::Zero();;
    Mat2x6 jacobian = Mat2x6::Zero();;

    uint32_t max_points_used_num = options_.kMaxSolvePointsNumber < p_w.size() ? options_.kMaxSolvePointsNumber : p_w.size();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        H.setZero();
        b.setZero();

        for (uint32_t i = 0; i < max_points_used_num; ++i) {
            Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            CONTINUE_IF(p_c.z() < options_.kMinValidDepth || std::isnan(p_c.z()));

            const float inv_depth = 1.0f / p_c.z();
            const float inv_depth2 = inv_depth * inv_depth;

            Vec2 residual = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_xy[i];

            Mat2x3 jacobian_2d_3d;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth2,
                              0, inv_depth, - p_c(1) * inv_depth2;
            jacobian.block<2, 3>(0, 0) = jacobian_2d_3d * (- q_wc.inverse().matrix());
            jacobian.block<2, 3>(0, 3) = jacobian_2d_3d * Utility::SkewSymmetricMatrix(p_c);

            switch (options_.kMethod) {
                default:
                case Method::kOptimize: {
                    H += jacobian.transpose() * jacobian;
                    b += - jacobian.transpose() * residual;
                    break;
                }
                case Method::kOptimizeHuber: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Huber(options_.kDefaultHuberKernelParameter, r_norm);
                    H += jacobian.transpose() * jacobian * kernel;
                    b += - jacobian.transpose() * residual * kernel;
                    break;
                }
                case Method::kOptimizeCauchy: {
                    const float r_norm = residual.norm();
                    const float kernel = this->Cauchy(options_.kDefaultCauchyKernelParameter, r_norm);
                    H += jacobian.transpose() * jacobian * kernel;
                    b += - jacobian.transpose() * residual * kernel;
                    break;
                }
            }
        }

        Vec6 dx = H.ldlt().solve(b);

        float norm_dx = dx.squaredNorm();
        RETURN_FALSE_IF(std::isnan(norm_dx) == true);
        norm_dx = std::sqrt(norm_dx);

        p_wc += dx.head<3>();
        q_wc = (q_wc * Utility::DeltaQ(dx.tail<3>())).normalized();

        BREAK_IF(norm_dx < options_.kMaxConvergeStep);
    }

    return true;
}

bool PnpSolver::EstimatePoseRansac(const std::vector<Vec3> &p_w,
                                   const std::vector<Vec2> &norm_xy,
                                   Quat &q_wc,
                                   Vec3 &p_wc,
                                   std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(p_w.size() != norm_xy.size() || p_w.empty());

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
            subPts2d.emplace_back(norm_xy[*it]);
        }

        // Compute pnp model with 3 points.
        q_wc = best_q_wc;
        p_wc = best_p_wc;
        EstimatePoseUseAll(subPts3d, subPts2d, q_wc, p_wc);

        // Apply pnp model on all points, statis inliers.
        score = 0;
        for (uint32_t i = 0; i < p_w.size(); ++i) {
            Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            Vec2 r = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_xy[i];
            if (r.squaredNorm() < options_.kMaxConvergeResidual) {
                ++score;
            }
        }

        if (score > best_score) {
            best_score = score;
            best_q_wc = q_wc;
            best_p_wc = p_wc;
        }

        BREAK_IF(float(score) / float(p_w.size()) > options_.kMinRansacInlierRatio);
    }

    CheckPnpStatus(p_w, norm_xy, q_wc, p_wc, status);
    return true;
}


bool PnpSolver::EstimatePoseDlt(const std::vector<Vec3> &p_w,
                                const std::vector<Vec2> &norm_xy,
                                Quat &q_wc,
                                Vec3 &p_wc,
                                std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(p_w.size() != norm_xy.size() || p_w.empty());

    Mat A = Mat::Zero(2 * p_w.size(), 12);
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        A.row(2 * i) << - p_w[i].x(), - p_w[i].y(), - p_w[i].z(), -1, 0, 0, 0, 0,
            norm_xy[i].x() * p_w[i].x(), norm_xy[i].x() * p_w[i].y(),
            norm_xy[i].x() * p_w[i].z(), norm_xy[i].x();
        A.row(2 * i + 1) << 0, 0, 0, 0, - p_w[i].x(), - p_w[i].y(), - p_w[i].z(), -1,
            norm_xy[i].y() * p_w[i].x(), norm_xy[i].y() * p_w[i].y(),
            norm_xy[i].y() * p_w[i].z(), norm_xy[i].y();
    }

    Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
    const Vec v = svd.matrixV().col(11);
    Mat3x4 P = Mat3x4::Zero();
    P << v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11);

    // Normalize rotation matrix.
    Mat3 R = P.block<3, 3>(0, 0);
    Eigen::JacobiSVD<Mat3> svd_r(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd_r.matrixU() * svd_r.matrixV().transpose();
    if (R.determinant() < 0) {
        R = - R;
    }

    // Compute translation.
    const float scale = 1.0f / svd_r.singularValues().mean();
    const Vec3 t = P.block<3, 1>(0, 3) * scale;

    q_wc = Quat(R).normalized();
    p_wc = - R.transpose() * t;

    status.resize(p_w.size(), static_cast<uint8_t>(Result::kSolved));
    return true;
}

void PnpSolver::CheckPnpStatus(const std::vector<Vec3> &p_w,
                               const std::vector<Vec2> &norm_xy,
                               Quat &q_wc,
                               Vec3 &p_wc,
                               std::vector<uint8_t> &status) {
    if (status.size() != norm_xy.size()) {
        status.resize(norm_xy.size(), static_cast<uint8_t>(Result::kUnsolved));
    }

    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(Result::kUnsolved) || status[i] == static_cast<uint8_t>(Result::kSolved)) {
            Vec3 p_c = q_wc.inverse() * (p_w[i] - p_wc);
            Vec2 r = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_xy[i];
            if (r.squaredNorm() >= options_.kMaxConvergeResidual) {
                status[i] = static_cast<uint8_t>(Result::kLargeResidual);
            } else {
                status[i] = static_cast<uint8_t>(Result::kSolved);
            }
        }
    }
}

}
