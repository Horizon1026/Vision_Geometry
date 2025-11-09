#include "geometry_epipolar.h"
#include "point_triangulator.h"
#include "slam_basic_math.h"
#include "slam_operations.h"

#include "slam_log_reporter.h"

#include <set>

namespace vision_geometry {

bool EpipolarSolver::EstimateEssential(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential,
                                       std::vector<uint8_t> &status) {
    switch (options_.kMethod) {
        case EpipolarMethod::kRansac: {
            return EstimateEssentialRansac(ref_norm_xy, cur_norm_xy, essential, status);
        }

        case EpipolarMethod::kUseAll: {
            return EstimateEssentialUseAll(ref_norm_xy, cur_norm_xy, essential, status);
        }

        default: {
            return false;
        }
    }
}

void EpipolarSolver::DecomposeEssentialMatrix(const Mat3 &essential, Mat3 &R0, Mat3 &R1, Vec3 &t0, Vec3 &t1) {
    Eigen::JacobiSVD<Mat3> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 Vt = svd.matrixV().transpose();

    // Make sure the determinants of U and V are both positive.
    if (U.determinant() < 0) {
        U = -U;
    }
    if (Vt.determinant() < 0) {
        Vt = -Vt;
    }

    // R means R(PI / 2).
    Mat3 R;
    R << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R0 = U * R * Vt;
    R1 = U * R.transpose() * Vt;
    t0 = U.col(2);
    t1 = -t0;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential,
                                             std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(EstimateEssentialUseAll(ref_norm_xy, cur_norm_xy, essential));

    RefineEssentialMatrix(essential);

    CheckEssentialPairsStatus(ref_norm_xy, cur_norm_xy, essential, status);

    return true;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential) {
    switch (options_.kModel) {
        case EpipolarModel::kFivePoints:
            RETURN_FALSE_IF_FALSE(EstimateEssentialUseFivePoints(ref_norm_xy, cur_norm_xy, essential));
            break;

        case EpipolarModel::kEightPoints:
        default:
            RETURN_FALSE_IF_FALSE(EstimateEssentialUseEightPoints(ref_norm_xy, cur_norm_xy, essential));
            break;
    }

    return true;
}

bool EpipolarSolver::EstimateEssentialRansac(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential,
                                             std::vector<uint8_t> &status) {
    if (ref_norm_xy.size() != cur_norm_xy.size() || ref_norm_xy.size() < 8) {
        return false;
    }

    // Preparation.
    uint32_t indice_size = 0;
    switch (options_.kModel) {
        case EpipolarModel::kFivePoints:
            indice_size = 5;
            break;
        case EpipolarModel::kEightPoints:
            indice_size = 8;
            break;
    }

    Mat3 best_essential = essential;
    Mat3 cur_essential = essential;
    uint32_t best_score = 0;
    uint32_t cur_score = 0;

    std::set<uint32_t> indice;
    std::vector<Vec2> sub_norm_xy_ref;
    std::vector<Vec2> sub_norm_xy_cur;
    sub_norm_xy_ref.reserve(indice_size);
    sub_norm_xy_cur.reserve(indice_size);

    std::vector<float> residuals;
    residuals.reserve(ref_norm_xy.size());

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Select samples.
        indice.clear();
        sub_norm_xy_ref.clear();
        sub_norm_xy_cur.clear();

        while (indice.size() < indice_size) {
            const uint32_t idx = std::rand() % ref_norm_xy.size();
            indice.insert(idx);
        }

        for (auto it = indice.cbegin(); it != indice.cend(); ++it) {
            sub_norm_xy_ref.emplace_back(ref_norm_xy[*it]);
            sub_norm_xy_cur.emplace_back(cur_norm_xy[*it]);
        }

        // Compute essential model.
        RETURN_FALSE_IF_FALSE(EstimateEssentialUseAll(sub_norm_xy_ref, sub_norm_xy_cur, cur_essential));

        // Apply essential model on all points, statis inliers.
        cur_score = 0;
        ComputeEssentialModelResidual(ref_norm_xy, cur_norm_xy, cur_essential, residuals);
        for (const float &residual: residuals) {
            if (residual < options_.kMaxEpipolarResidual) {
                ++cur_score;
            }
        }

        if (cur_score > best_score) {
            best_score = cur_score;
            best_essential = cur_essential;
        }

        if (static_cast<float>(cur_score) / static_cast<float>(ref_norm_xy.size()) > options_.kMinRansacInlierRatio) {
            break;
        }
    }

    essential = best_essential;

    CheckEssentialPairsStatus(ref_norm_xy, cur_norm_xy, essential, status);

    return true;
}

void EpipolarSolver::ComputeEssentialModelResidual(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential,
                                                   std::vector<float> &residuals) {
    if (residuals.size() != ref_norm_xy.size()) {
        residuals.resize(ref_norm_xy.size());
    }

    for (uint32_t i = 0; i < ref_norm_xy.size(); ++i) {
        const Vec3 pts_ref = Vec3(ref_norm_xy[i].x(), ref_norm_xy[i].y(), 1.0f);
        const Vec3 pts_cur = Vec3(cur_norm_xy[i].x(), cur_norm_xy[i].y(), 1.0f);
        residuals[i] = pts_cur.transpose() * essential * pts_ref;
        residuals[i] = std::fabs(residuals[i]);
    }
}

float EpipolarSolver::ComputeEssentialModelResidualSummary(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential) {
    std::vector<float> residuals;
    residuals.resize(ref_norm_xy.size());
    ComputeEssentialModelResidual(ref_norm_xy, cur_norm_xy, essential, residuals);

    float sum_residual = 0.0f;
    for (const float &residual: residuals) {
        sum_residual += residual;
    }
    return sum_residual / static_cast<float>(ref_norm_xy.size());
}

void EpipolarSolver::CheckEssentialPairsStatus(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential,
                                               std::vector<uint8_t> &status) {
    if (status.size() != ref_norm_xy.size()) {
        status.resize(ref_norm_xy.size(), static_cast<uint8_t>(EpipolarResult::kUnsolved));
    }

    // Check those features that haven't been solved.
    std::vector<float> residuals;
    residuals.reserve(ref_norm_xy.size());
    ComputeEssentialModelResidual(ref_norm_xy, cur_norm_xy, essential, residuals);
    for (uint32_t i = 0; i < ref_norm_xy.size(); ++i) {
        if (status[i] == static_cast<uint8_t>(EpipolarResult::kUnsolved) || status[i] == static_cast<uint8_t>(EpipolarResult::kSolved)) {
            if (residuals[i] < options_.kMaxEpipolarResidual) {
                status[i] = static_cast<uint8_t>(EpipolarResult::kSolved);
            } else {
                status[i] = static_cast<uint8_t>(EpipolarResult::kLargeResidual);
            }
        }
    }
}

bool EpipolarSolver::RecoverPoseFromEssential(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential, Mat3 &R_cr,
                                              Vec3 &t_cr) {
    if (ref_norm_xy.size() != cur_norm_xy.size() || ref_norm_xy.empty()) {
        return false;
    }

    // Decompose essential matrix. R, t, q means R_cr, t_cr, q_cr.
    // R_cr means rotation from ref to cur.
    Mat3 R0, R1;
    Vec3 t0, t1;
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
    PointTriangulator solver;
    solver.options().kMethod = PointTriangulator::Method::kAnalytic;

    for (Statis &item: statis) {
        std::vector<Quat> q_rc = {Quat::Identity(), item.q_cr.inverse()};
        std::vector<Vec3> p_rc = {Vec3::Identity(), -Vec3(item.q_cr.inverse() * item.p_cr)};

        for (uint32_t i = 0; i < ref_norm_xy.size(); ++i) {
            std::vector<Vec2> norm_xy = {ref_norm_xy[i], cur_norm_xy[i]};

            // Check is point.z positive in ref frame.
            Vec3 p_r;
            solver.Triangulate(q_rc, p_rc, norm_xy, p_r);
            if (p_r.z() > kZeroFloat) {
                // Check is point.z positive in cur frame.
                Vec3 p_c = item.q_cr * p_r + item.p_cr;
                if (p_c.z() > kZeroFloat) {
                    ++item.score;
                }
            }
        }
    }

    // Select the most correct pose.
    uint32_t max_score = 0;
    for (Statis &item: statis) {
        if (item.score >= max_score) {
            R_cr = Mat3(item.q_cr);
            t_cr = item.p_cr;
            max_score = item.score;
        }
    }

    return true;
}

}  // namespace vision_geometry
