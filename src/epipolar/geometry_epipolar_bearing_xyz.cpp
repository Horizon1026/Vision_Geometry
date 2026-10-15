#include "geometry_epipolar.h"
#include "point_triangulator.h"

#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include "algorithm"
#include "limits"
#include "random"
#include "set"

namespace vision_geometry {

bool EpipolarSolver::EstimateEssential(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential,
                                       std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size());
    switch (options_.kMethod) {
        default:
        case EpipolarMethod::kRansac: {
            return EstimateEssentialRansac(ref_bearing_xyz, cur_bearing_xyz, essential, status);
        }

        case EpipolarMethod::kUseAll: {
            for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
                RETURN_FALSE_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                                ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
            }
            return EstimateEssentialUseAll(ref_bearing_xyz, cur_bearing_xyz, essential, status);
        }
    }
    return false;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential,
                                             std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(EstimateEssentialUseAll(ref_bearing_xyz, cur_bearing_xyz, essential));
    RefineEssentialMatrix(essential);
    CheckEssentialPairsStatus(ref_bearing_xyz, cur_bearing_xyz, essential, status);
    return true;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential) {
    switch (options_.kModel) {
        case EpipolarModel::kFivePoints:
            RETURN_FALSE_IF_FALSE(EstimateEssentialUseFivePoints(ref_bearing_xyz, cur_bearing_xyz, essential));
            break;

        case EpipolarModel::kEightPoints:
        default:
            RETURN_FALSE_IF_FALSE(EstimateEssentialUseEightPoints(ref_bearing_xyz, cur_bearing_xyz, essential));
            break;
    }

    return true;
}

bool EpipolarSolver::EstimateEssentialRansac(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential,
                                            std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(!status.empty() && status.size() != ref_bearing_xyz.size());
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        if (status.empty() || status[i] == static_cast<uint8_t>(EpipolarResult::kSolved)) {
            RETURN_FALSE_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                            ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
            indices.push_back(i);
        }
    }
    const uint32_t primary_sample_size = options_.kModel == EpipolarModel::kFivePoints ? 5 : 8;
    RETURN_FALSE_IF(indices.size() < primary_sample_size);
    Mat3 best_essential = Mat3::Zero();
    uint32_t best_score = 0;
    std::mt19937 rng(std::random_device{}());
    // Retry with the eight-point model if the five-point model has insufficient support.
    for (int pass = 0; pass < 2; ++pass) {
        const uint32_t sample_size = pass == 0 ? primary_sample_size : 8;
        BREAK_IF(pass == 1 && (primary_sample_size != 5 || indices.size() < 8 || best_score > indices.size() / 2));
        uint32_t max_iter = options_.kMaxIteration;
        for (uint32_t iter = 0; iter < max_iter; ++iter) {
            std::shuffle(indices.begin(), indices.end(), rng);
            std::vector<Vec3> ref;
            std::vector<Vec3> cur;
            for (uint32_t j = 0; j < sample_size; ++j) {
                ref.push_back(ref_bearing_xyz[indices[j]]);
                cur.push_back(cur_bearing_xyz[indices[j]]);
            }
            Mat3 candidate = Mat3::Zero();
            const bool solved = sample_size == 5 ? EstimateEssentialUseFivePoints(ref, cur, candidate) :
                                                  EstimateEssentialUseEightPoints(ref, cur, candidate);
            CONTINUE_IF(!solved || !candidate.allFinite());
            RefineEssentialMatrix(candidate);
            uint32_t score = 0;
            std::vector<float> residuals;
            std::vector<Vec3> selected_ref;
            std::vector<Vec3> selected_cur;
            selected_ref.reserve(indices.size());
            selected_cur.reserve(indices.size());
            for (const auto i: indices) {
                selected_ref.push_back(ref_bearing_xyz[i]);
                selected_cur.push_back(cur_bearing_xyz[i]);
            }
            ComputeEssentialModelResidual(selected_ref, selected_cur, candidate, residuals);
            for (const auto residual: residuals) {
                score += residual < options_.kMaxEpipolarResidual;
            }
            if (score > best_score) {
                best_score = score;
                best_essential = candidate;
            }
            const float ratio = static_cast<float>(best_score) / indices.size();
            BREAK_IF(ratio > options_.kMinRansacInlierRatio);
            const float probability = std::pow(ratio, static_cast<float>(sample_size));
            if (probability > 0.0f && probability < 1.0f) {
                const float required = std::ceil(std::log(0.01f) / std::log1p(-probability));
                if (std::isfinite(required) && required < static_cast<float>(max_iter)) {
                    max_iter = std::max(iter + 1, static_cast<uint32_t>(required));
                }
            }
        }
    }
    RETURN_FALSE_IF(best_score < primary_sample_size);
    essential = best_essential;
    if (status.empty()) {
        status.assign(ref_bearing_xyz.size(), static_cast<uint8_t>(EpipolarResult::kUnsolved));
    }
    std::vector<float> residuals;
    ComputeEssentialModelResidual(ref_bearing_xyz, cur_bearing_xyz, essential, residuals);
    for (const auto i: indices) {
        status[i] = static_cast<uint8_t>(residuals[i] < options_.kMaxEpipolarResidual ? EpipolarResult::kSolved : EpipolarResult::kLargeResidual);
    }
    return true;
}

void EpipolarSolver::ComputeEssentialModelResidual(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const Mat3 &essential,
                                                   std::vector<float> &residuals) {
    if (residuals.size() != ref_bearing_xyz.size()) {
        residuals.resize(ref_bearing_xyz.size());
    }

    for (uint32_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        const Vec3 pts_ref = ref_bearing_xyz[i].normalized();
        const Vec3 pts_cur = cur_bearing_xyz[i].normalized();

        // Epipolar line in the current frame: l' = E * pts_ref.
        const Vec3 line_cur = essential * pts_ref;
        // Epipolar line in the reference frame: l = E^T * pts_cur.
        const Vec3 line_ref = essential.transpose() * pts_cur;

        // Algebraic error (signed distance from point to line in homogeneous coords).
        const float err = pts_cur.transpose() * essential * pts_ref;

        // Mean sine of the angles to the two epipolar planes (unit sphere).
        const float norm_ref = line_ref.norm();
        const float norm_cur = line_cur.norm();
        if (norm_ref > kZeroFloat && norm_cur > kZeroFloat) {
            residuals[i] = std::fabs(err) * (1.0f / norm_ref + 1.0f / norm_cur) * 0.5f;
        } else {
            residuals[i] = INFINITY;
        }
    }
}

float EpipolarSolver::ComputeEssentialModelResidualSummary(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const Mat3 &essential) {
    std::vector<float> residuals;
    residuals.resize(ref_bearing_xyz.size());
    ComputeEssentialModelResidual(ref_bearing_xyz, cur_bearing_xyz, essential, residuals);

    float sum_residual = 0.0f;
    for (const float &residual: residuals) {
        sum_residual += residual;
    }
    return sum_residual / static_cast<float>(ref_bearing_xyz.size());
}

void EpipolarSolver::CheckEssentialPairsStatus(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential,
                                               std::vector<uint8_t> &status) {
    if (status.size() != ref_bearing_xyz.size()) {
        status.assign(ref_bearing_xyz.size(), static_cast<uint8_t>(EpipolarResult::kUnsolved));
    }

    // Check those features that haven't been solved.
    std::vector<float> residuals;
    residuals.reserve(ref_bearing_xyz.size());
    ComputeEssentialModelResidual(ref_bearing_xyz, cur_bearing_xyz, essential, residuals);
    for (uint32_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(EpipolarResult::kUnsolved) || status[i] == static_cast<uint8_t>(EpipolarResult::kSolved)) {
            if (residuals[i] < options_.kMaxEpipolarResidual) {
                status[i] = static_cast<uint8_t>(EpipolarResult::kSolved);
            } else {
                status[i] = static_cast<uint8_t>(EpipolarResult::kLargeResidual);
            }
        }
    }
}

bool EpipolarSolver::RecoverPoseFromEssential(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const Mat3 &essential, Mat3 &R_cr,
                                              Vec3 &t_cr) {
    RETURN_FALSE_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size() || ref_bearing_xyz.empty());

    RETURN_FALSE_IF(!essential.allFinite() || essential.norm() <= kZeroFloat);
    for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        RETURN_FALSE_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                        ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
    }

    // Decompose essential matrix. R, t, q means R_cr, t_cr, q_cr.
    Mat3 R0 = Mat3::Zero();
    Mat3 R1 = Mat3::Zero();
    Vec3 t0 = Vec3::Zero();
    Vec3 t1 = Vec3::Zero();
    DecomposeEssentialMatrix(essential, R0, R1, t0, t1);
    Quat q0(R0), q1(R1);
    q0.normalize();
    q1.normalize();

    // Construct 4 different situations.
    struct Statis {
        uint32_t score = 0;
        Quat q_cr = Quat::Identity();
        Vec3 p_cr = Vec3::Zero();
    };
    std::array<Statis, 4> statis = {
        Statis {0, q0, t0},
        Statis {0, q0, t1},
        Statis {0, q1, t0},
        Statis {0, q1, t1},
    };

    // Triangulization points to find which pose is right.
    PointTriangulator solver = {};
    solver.options().kMethod = PointTriangulator::Method::kAnalytic;

    for (Statis &item: statis) {
        std::vector<Quat> q_rc = {Quat::Identity(), item.q_cr.inverse()};
        std::vector<Vec3> p_rc = {Vec3::Zero(), -Vec3(item.q_cr.inverse() * item.p_cr)};

        for (uint32_t i = 0; i < ref_bearing_xyz.size(); ++i) {
            std::vector<Vec3> bearing_xyz = {ref_bearing_xyz[i].normalized(), cur_bearing_xyz[i].normalized()};

            // Check positive range along the reference ray.
            Vec3 p_r = Vec3::Zero();
            CONTINUE_IF(!solver.Triangulate(p_rc, q_rc, bearing_xyz, p_r));
            if (p_r.dot(bearing_xyz[0]) > kZeroFloat) {
                // Check positive range along the current ray.
                Vec3 p_c = item.q_cr * p_r + item.p_cr;
                if (p_c.dot(bearing_xyz[1]) > kZeroFloat) {
                    ++item.score;
                }
            }
        }
    }

    // Select the pose with positive range along both observed rays.
    uint32_t max_score = 0;
    for (Statis &item: statis) {
        if (item.score >= max_score) {
            R_cr = Mat3(item.q_cr);
            t_cr = item.p_cr;
            max_score = item.score;
        }
    }

    return max_score > 0;
}

bool EpipolarSolver::EstimateEssentialUseEightPoints(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential) {
    RETURN_FALSE_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size() || ref_bearing_xyz.size() < 8);

    const uint32_t rows = std::min(options_.kMaxSolvePointsNumber, static_cast<uint32_t>(ref_bearing_xyz.size()));
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, 9);

    // Construct eight point model Ae = 0.
    for (uint32_t i = 0; i < rows; ++i) {
        const Vec3 ref = ref_bearing_xyz[i].normalized();
        const Vec3 cur = cur_bearing_xyz[i].normalized();
        A.row(i) << cur.x() * ref.transpose(), cur.y() * ref.transpose(), cur.z() * ref.transpose();
    }

    RETURN_FALSE_IF(rows < 8);

    return SolveEightPointConstraint(A, essential, true);
}

bool EpipolarSolver::EstimateEssentialUseFivePoints(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential) {
    if (ref_bearing_xyz.size() != cur_bearing_xyz.size() || ref_bearing_xyz.size() < 5) {
        ReportError("[EpipolarSolver] Failed to estimate essential matrix. Invalid input size: ref_bearing_xyz.size() = "
                    << ref_bearing_xyz.size() << ", cur_bearing_xyz.size() = " << cur_bearing_xyz.size());
        return false;
    }

    // Storage all possible results.
    std::vector<Mat3> all_essentials;
    all_essentials.reserve(10);

    // Construct eight point model Qe = 0.
    Eigen::Matrix<float, Eigen::Dynamic, 9> Q = Eigen::Matrix<float, Eigen::Dynamic, 9>::Zero(cur_bearing_xyz.size(), 9);
    for (uint32_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        const Vec3 ref = ref_bearing_xyz[i].normalized();
        const Vec3 cur = cur_bearing_xyz[i].normalized();
        Q.row(i) << cur.x() * ref.transpose(), cur.y() * ref.transpose(), cur.z() * ref.transpose();
    }

    RETURN_FALSE_IF_FALSE(SolveFivePointConstraint(Q, all_essentials));

    return FindBestOneFromAllPossibleEssentials(ref_bearing_xyz, cur_bearing_xyz, all_essentials, essential);
}

bool EpipolarSolver::FindBestOneFromAllPossibleEssentials(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz,
                                                          const std::vector<Mat3> &essentials, Mat3 &best_essential) {
    RETURN_FALSE_IF(essentials.empty());

    float min_residual = INFINITY;
    std::vector<float> residuals;
    residuals.reserve(ref_bearing_xyz.size());

    for (const Mat3 &essential: essentials) {
        float sum = 0.0f;
        ComputeEssentialModelResidual(ref_bearing_xyz, cur_bearing_xyz, essential, residuals);
        for (const float &residual: residuals) {
            sum += residual;
        }

        if (sum < min_residual) {
            min_residual = sum;
            best_essential = essential;
        }
    }

    return std::isfinite(min_residual);
}

}  // namespace vision_geometry
