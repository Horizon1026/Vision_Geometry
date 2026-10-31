#include "geometry_pnp.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

#include "algorithm"
#include "limits"
#include "random"

namespace vision_geometry {

// Euclidean distance on the unit sphere: 2 * sin(angle / 2).
float PnpSolver::BearingError(const Vec3 &p_c, const Vec3 &bearing) const {
    if (!p_c.allFinite() || p_c.norm() <= kZeroFloat) {
        return std::numeric_limits<float>::infinity();
    }
    return (p_c.normalized() - bearing.normalized()).norm();
}

bool PnpSolver::EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(p_w.empty() || p_w.size() != bearing_xyz.size());
    RETURN_FALSE_IF(!status.empty() && status.size() != p_w.size());
    if (options_.kMethod == Method::kDirectLinearTransform) {
        RETURN_FALSE_IF_FALSE(EstimatePoseDlt(p_w, bearing_xyz, p_wc, q_wc));
        CheckPnpStatus(p_w, bearing_xyz, p_wc, q_wc, status);
        return true;
    }
    return EstimatePose(p_w, std::vector<Vec3>(p_w.size(), Vec3::Zero()), std::vector<Quat>(p_w.size(), Quat::Identity()), bearing_xyz, p_wc, q_wc, status);
}

bool PnpSolver::OptimizeBearings(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic,
                                 const std::vector<Vec3> &bearing_xyz, Vec3 &p_wi, Quat &q_wi) {
    const uint32_t count = std::min<uint32_t>(options_.kMaxSolvePointsNumber, p_w.size());
    RETURN_FALSE_IF(count < 3 || options_.kMaxIteration == 0);
    q_wi.normalize();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        Mat6 H = Mat6::Zero();
        Vec6 b = Vec6::Zero();
        uint32_t used = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const Vec3 p_i = q_wi.inverse() * (p_w[i] - p_wi);
            const Vec3 p_c = q_ic[i].inverse() * (p_i - p_ic[i]);
            const float range = p_c.norm();
            CONTINUE_IF(!p_c.allFinite() || range <= options_.kMinValidDepth);
            ++used;
            const Vec3 predicted = p_c / range;
            const Vec3 residual = predicted - bearing_xyz[i].normalized();
            const Mat3 projection = (Mat3::Identity() - predicted * predicted.transpose()) / range;
            Eigen::Matrix<float, 3, 6> jacobian = Eigen::Matrix<float, 3, 6>::Zero();
            jacobian.leftCols<3>() = -projection * (q_wi * q_ic[i]).inverse().matrix();
            jacobian.rightCols<3>() = projection * q_ic[i].inverse().matrix() * Utility::SkewSymmetricMatrix(p_i);
            float weight = 1.0f;
            if (options_.kMethod == Method::kOptimizeHuber) {
                weight = Huber(options_.kDefaultHuberKernelParameter, residual.norm());
            } else if (options_.kMethod == Method::kOptimizeCauchy) {
                weight = Cauchy(options_.kDefaultCauchyKernelParameter, residual.norm());
            }
            H += weight * jacobian.transpose() * jacobian;
            b -= weight * jacobian.transpose() * residual;
        }
        RETURN_FALSE_IF(used < 3);
        Eigen::LDLT<Mat6> decomposition(H);
        RETURN_FALSE_IF(decomposition.info() != Eigen::Success ||
                        decomposition.vectorD().cwiseAbs().minCoeff() <= std::numeric_limits<float>::epsilon() * H.diagonal().cwiseAbs().maxCoeff());
        const Vec6 dx = decomposition.solve(b);
        RETURN_FALSE_IF(!dx.allFinite());
        p_wi += dx.head<3>();
        q_wi = (q_wi * Utility::DeltaQ(dx.tail<3>())).normalized();
        BREAK_IF(dx.norm() < options_.kMaxConvergeStep);
    }
    return p_wi.allFinite() && q_wi.coeffs().allFinite();
}

bool PnpSolver::EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec3> &bearing_xyz,
                             Vec3 &p_wi, Quat &q_wi, std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(p_w.empty() || p_w.size() != bearing_xyz.size() || p_w.size() < 3 || q_ic.size() != p_w.size() || p_ic.size() != p_w.size());
    RETURN_FALSE_IF(!status.empty() && status.size() != p_w.size());
    RETURN_FALSE_IF(!q_wi.coeffs().allFinite() || q_wi.norm() <= kZeroFloat || !p_wi.allFinite());
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        RETURN_FALSE_IF(!q_ic[i].coeffs().allFinite() || std::fabs(q_ic[i].norm() - 1.0f) > 1e-3f || !p_ic[i].allFinite());
    }
    if (options_.kMethod == Method::kOptimize || options_.kMethod == Method::kOptimizeHuber || options_.kMethod == Method::kOptimizeCauchy) {
        RETURN_FALSE_IF_FALSE(OptimizeBearings(p_w, p_ic, q_ic, bearing_xyz, p_wi, q_wi));
        // As in the Vec2 API, only classify previously unsolved entries here.
        if (status.empty()) {
            status.assign(p_w.size(), static_cast<uint8_t>(Result::kUnsolved));
        }
        for (uint32_t i = 0; i < p_w.size(); ++i) {
            CONTINUE_IF(status[i] != static_cast<uint8_t>(Result::kUnsolved));
            const Vec3 p_c = q_ic[i].inverse() * (q_wi.inverse() * (p_w[i] - p_wi) - p_ic[i]);
            status[i] = static_cast<uint8_t>(BearingError(p_c, bearing_xyz[i]) < options_.kMaxPnpResidual ? Result::kSolved : Result::kLargeResidual);
        }
        return true;
    }
    RETURN_FALSE_IF(options_.kMethod != Method::kOptimizeRansac);
    std::vector<uint32_t> selected;
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        if (status.empty() || status[i] == static_cast<uint8_t>(Result::kSolved)) {
            selected.push_back(i);
        }
    }
    RETURN_FALSE_IF(selected.size() < 3);
    Quat best_q = q_wi;
    Vec3 best_p = p_wi;
    uint32_t best_score = 0;
    std::mt19937 rng(std::random_device {}());
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        std::shuffle(selected.begin(), selected.end(), rng);
        std::vector<Vec3> points;
        std::vector<Vec3> bearings;
        std::vector<Vec3> translations;
        std::vector<Quat> rotations;
        for (uint32_t j = 0; j < 3; ++j) {
            const uint32_t i = selected[j];
            points.push_back(p_w[i]);
            bearings.push_back(bearing_xyz[i]);
            translations.push_back(p_ic[i]);
            rotations.push_back(q_ic[i]);
        }
        Quat candidate_q = best_q;
        Vec3 candidate_p = best_p;
        CONTINUE_IF(!OptimizeBearings(points, translations, rotations, bearings, candidate_p, candidate_q));
        uint32_t score = 0;
        for (const uint32_t i: selected) {
            const Vec3 p_c = q_ic[i].inverse() * (candidate_q.inverse() * (p_w[i] - candidate_p) - p_ic[i]);
            const float error = BearingError(p_c, bearing_xyz[i]);
            if (error * error < options_.kMaxConvergeResidual) {
                ++score;
            }
        }
        if (score > best_score) {
            best_score = score;
            best_q = candidate_q;
            best_p = candidate_p;
        }
        BREAK_IF(static_cast<float>(best_score) / selected.size() > options_.kMinRansacInlierRatio);
    }
    RETURN_FALSE_IF(best_score < 3);
    q_wi = best_q;
    p_wi = best_p;
    if (status.empty()) {
        status.assign(p_w.size(), static_cast<uint8_t>(Result::kUnsolved));
    }
    for (const uint32_t i: selected) {
        const Vec3 p_c = q_ic[i].inverse() * (q_wi.inverse() * (p_w[i] - p_wi) - p_ic[i]);
        const float error = BearingError(p_c, bearing_xyz[i]);
        status[i] = static_cast<uint8_t>(error * error < options_.kMaxConvergeResidual ? Result::kSolved : Result::kLargeResidual);
    }
    return true;
}

bool PnpSolver::EstimatePoseDlt(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wc, Quat &q_wc) {
    RETURN_FALSE_IF(p_w.size() < 6);
    Mat A = Mat::Zero(3 * p_w.size(), 12);
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        const Mat3 skew = Utility::SkewSymmetricMatrix(Vec3(bearing_xyz[i].normalized()));
        const Vec4 point(p_w[i].x(), p_w[i].y(), p_w[i].z(), 1.0f);
        for (int col = 0; col < 3; ++col) {
            A.block<3, 4>(3 * i, 4 * col) = skew.col(col) * point.transpose();
        }
    }
    Mat3x4 P = Mat3x4::Zero();
    RETURN_FALSE_IF_FALSE(SolveDltConstraint(A, P, true));
    Mat3 R_cw = P.leftCols<3>();
    if (R_cw.determinant() < 0.0f) {
        P = -P;
        R_cw = -R_cw;
    }
    Eigen::JacobiSVD<Mat3> svd_r(R_cw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const float scale = svd_r.singularValues().mean();
    RETURN_FALSE_IF(scale <= kZeroFloat);
    R_cw = svd_r.matrixU() * svd_r.matrixV().transpose();
    const Vec3 t_cw = P.col(3) / scale;
    q_wc = Quat(R_cw.transpose()).normalized();
    p_wc = -R_cw.transpose() * t_cw;
    return q_wc.coeffs().allFinite() && p_wc.allFinite();
}

void PnpSolver::CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status) {
    CheckPnpStatus(p_w, std::vector<Vec3>(p_w.size(), Vec3::Zero()), std::vector<Quat>(p_w.size(), Quat::Identity()), bearing_xyz, p_wc, q_wc, status);
}

void PnpSolver::CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec3> &bearing_xyz,
                               Vec3 &p_wi, Quat &q_wi, std::vector<uint8_t> &status) {
    RETURN_IF(p_w.empty() || p_w.size() != bearing_xyz.size() || q_ic.size() != p_w.size() || p_ic.size() != p_w.size());
    if (status.size() != p_w.size()) {
        status.assign(p_w.size(), static_cast<uint8_t>(Result::kUnsolved));
    }
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        CONTINUE_IF(status[i] != static_cast<uint8_t>(Result::kUnsolved) && status[i] != static_cast<uint8_t>(Result::kSolved));
        const Vec3 p_c = q_ic[i].inverse() * (q_wi.inverse() * (p_w[i] - p_wi) - p_ic[i]);
        const float error = BearingError(p_c, bearing_xyz[i]);
        status[i] = static_cast<uint8_t>(error * error < options_.kMaxConvergeResidual ? Result::kSolved : Result::kLargeResidual);
    }
}

float PnpSolver::ComputeAverageReprojectionError(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, const Vec3 &p_wc, const Quat &q_wc) const {
    if (p_w.empty() || p_w.size() != bearing_xyz.size()) {
        return -1.0f;
    }
    float total = 0.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < p_w.size(); ++i) {
        const float error = BearingError(q_wc.inverse() * (p_w[i] - p_wc), bearing_xyz[i]);
        if (std::isfinite(error)) {
            total += error;
            ++count;
        }
    }
    return count ? total / count : -1.0f;
}

}  // namespace vision_geometry
