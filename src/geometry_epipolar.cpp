#include "geometry_epipolar.h"
#include "geometry_triangulation.h"
#include "math_kinematics.h"
#include "slam_operations.h"

#include "log_api.h"

#include <set>

namespace VISION_GEOMETRY {

bool EpipolarSolver::EstimateEssential(const std::vector<Vec2> &norm_uv_ref,
                                       const std::vector<Vec2> &norm_uv_cur,
                                       Mat3 &essential,
                                       std::vector<EpipolarResult> &status) {
    switch (options_.kMethod) {
        case EpipolarMethod::EPIPOLAR_RANSAC: {
            return EstimateEssentialRansac(norm_uv_ref, norm_uv_cur, essential, status);
        }

        case EpipolarMethod::EPIPOLAR_ALL:
        case EpipolarMethod::EPIPOLAR_HUBER:
        case EpipolarMethod::EPIPOLAR_CAUCHY: {
            return EstimateEssentialUseAll(norm_uv_ref, norm_uv_cur, essential, status);
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
        U = - U;
    }
    if (Vt.determinant() < 0) {
        Vt = - Vt;
    }

    // R means R(PI / 2).
    Mat3 R;
    R << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R0 = U * R * Vt;
    R1 = U * R.transpose() * Vt;
    t0 = U.col(2);
    t1 = - t0;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                             const std::vector<Vec2> &norm_uv_cur,
                                             Mat3 &essential,
                                             std::vector<EpipolarResult> &status) {
    RETURN_FALSE_IF_FALSE(EstimateEssentialUseAll(norm_uv_ref, norm_uv_cur, essential));

    RefineEssentialMatrix(essential);

    CheckEssentialPairsStatus(norm_uv_ref, norm_uv_cur, essential, status);

    return true;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                             const std::vector<Vec2> &norm_uv_cur,
                                             Mat3 &essential) {
    switch (options_.kModel) {
        case EpipolarModel::FIVE_POINTS:
            RETURN_FALSE_IF_FALSE(EstimateEssentialUseFivePoints(norm_uv_ref, norm_uv_cur, essential));
            break;

        case EpipolarModel::EIGHT_POINTS:
        default:
            RETURN_FALSE_IF_FALSE(EstimateEssentialUseEightPoints(norm_uv_ref, norm_uv_cur, essential));
            break;
    }

    return true;
}

bool EpipolarSolver::EstimateEssentialRansac(const std::vector<Vec2> &norm_uv_ref,
                                             const std::vector<Vec2> &norm_uv_cur,
                                             Mat3 &essential,
                                             std::vector<EpipolarResult> &status) {
    if (norm_uv_ref.size() != norm_uv_cur.size() || norm_uv_ref.size() < 8) {
        return false;
    }

    // Preparation.
    uint32_t indice_size = 0;
    switch (options_.kModel) {
        case EpipolarModel::FIVE_POINTS:
            indice_size = 5;
            break;
        case EpipolarModel::EIGHT_POINTS:
            indice_size = 8;
            break;
    }

    Mat3 best_essential = essential;
    Mat3 cur_essential = essential;
    uint32_t best_score = 0;
    uint32_t cur_score = 0;

    std::set<uint32_t> indice;
    std::vector<Vec2> sub_norm_uv_ref;
    std::vector<Vec2> sub_norm_uv_cur;
    sub_norm_uv_ref.reserve(indice_size);
    sub_norm_uv_cur.reserve(indice_size);

    std::vector<float> residuals;
    residuals.reserve(norm_uv_ref.size());

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Select samples.
        indice.clear();
        sub_norm_uv_ref.clear();
        sub_norm_uv_cur.clear();

        while (indice.size() < indice_size) {
            const uint32_t idx = std::rand() % norm_uv_ref.size();
            indice.insert(idx);
        }

        for (auto it = indice.cbegin(); it != indice.cend(); ++it) {
            sub_norm_uv_ref.emplace_back(norm_uv_ref[*it]);
            sub_norm_uv_cur.emplace_back(norm_uv_cur[*it]);
        }

        // Compute essential model.
        RETURN_FALSE_IF_FALSE(EstimateEssentialUseAll(sub_norm_uv_ref, sub_norm_uv_cur, cur_essential));

        // Apply essential model on all points, statis inliers.
        cur_score = 0;
        ComputeEssentialModelResidual(norm_uv_ref, norm_uv_cur, cur_essential, residuals);
        for (const float &residual : residuals) {
            if (residual < options_.kMaxEpipolarResidual) {
                ++cur_score;
            }
        }

        if (cur_score > best_score) {
            best_score = cur_score;
            best_essential = cur_essential;
        }

        if (static_cast<float>(cur_score) / static_cast<float>(norm_uv_ref.size()) > options_.kMinRansacInlierRatio) {
            break;
        }
    }

    essential = best_essential;

    CheckEssentialPairsStatus(norm_uv_ref, norm_uv_cur, essential, status);

    return true;
}

void EpipolarSolver::ComputeEssentialModelResidual(const std::vector<Vec2> &norm_uv_ref,
                                                   const std::vector<Vec2> &norm_uv_cur,
                                                   const Mat3 &essential,
                                                   std::vector<float> &residuals) {
    if (residuals.size() != norm_uv_ref.size()) {
        residuals.resize(norm_uv_ref.size());
    }

    for (uint32_t i = 0; i < norm_uv_ref.size(); ++i) {
        const Vec3 pts_ref = Vec3(norm_uv_ref[i].x(), norm_uv_ref[i].y(), 1.0f);
        const Vec3 pts_cur = Vec3(norm_uv_cur[i].x(), norm_uv_cur[i].y(), 1.0f);
        residuals[i] = pts_cur.transpose() * essential * pts_ref;
        residuals[i] = std::fabs(residuals[i]);
    }
}

float EpipolarSolver::ComputeEssentialModelResidualSummary(const std::vector<Vec2> &norm_uv_ref,
                                                           const std::vector<Vec2> &norm_uv_cur,
                                                           const Mat3 &essential) {
    std::vector<float> residuals;
    residuals.resize(norm_uv_ref.size());
    ComputeEssentialModelResidual(norm_uv_ref, norm_uv_cur, essential, residuals);

    float sum_residual = 0.0f;
    for (const float &residual : residuals) {
        sum_residual += residual;
    }
    return sum_residual / static_cast<float>(norm_uv_ref.size());
}

void EpipolarSolver::CheckEssentialPairsStatus(const std::vector<Vec2> &norm_uv_ref,
                                               const std::vector<Vec2> &norm_uv_cur,
                                               Mat3 &essential,
                                               std::vector<EpipolarResult> &status) {
    if (status.size() != norm_uv_ref.size()) {
        status.resize(norm_uv_ref.size(), EpipolarResult::UNSOLVED);
    }

    // Check those features that haven't been solved.
    std::vector<float> residuals;
    residuals.reserve(norm_uv_ref.size());
    ComputeEssentialModelResidual(norm_uv_ref, norm_uv_cur, essential, residuals);
    for (uint32_t i = 0; i < norm_uv_ref.size(); ++i) {
        if (status[i] == EpipolarResult::UNSOLVED) {
            if (residuals[i] < options_.kMaxEpipolarResidual) {
                status[i] = EpipolarResult::SOLVED;
            } else {
                status[i] = EpipolarResult::LARGE_RISIDUAL;
            }
        }
    }
}

bool EpipolarSolver::RecoverPoseFromEssential(const std::vector<Vec2> &norm_uv_ref,
                                              const std::vector<Vec2> &norm_uv_cur,
                                              const Mat3 &essential,
                                              Mat3 &R_cr,
                                              Vec3 &t_cr) {
    if (norm_uv_ref.size() != norm_uv_cur.size() || norm_uv_ref.empty()) {
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
        Statis{ 0, q0, t0 },
        Statis{ 0, q0, t1 },
        Statis{ 0, q1, t0 },
        Statis{ 0, q1, t1 },
    };

    // Triangulization points to find which pose is right.
    Triangulator solver;
    solver.options().kMethod = Triangulator::TriangulationMethod::ANALYTIC;

    for (Statis &item : statis) {
        std::vector<Quat> q_rc = { Quat::Identity(), item.q_cr.inverse() };
        std::vector<Vec3> p_rc = { Vec3::Identity(), - Vec3(item.q_cr.inverse() * item.p_cr) };

        for (uint32_t i = 0; i < norm_uv_ref.size(); ++i) {
            std::vector<Vec2> norm_uv = { norm_uv_ref[i], norm_uv_cur[i] };

            // Check is point.z positive in ref frame.
            Vec3 p_r;
            solver.Triangulate(q_rc, p_rc, norm_uv, p_r);
            if (p_r.z() > kZero) {
                // Check is point.z positive in cur frame.
                Vec3 p_c = item.q_cr * p_r + item.p_cr;
                if (p_c.z() > kZero) {
                    ++item.score;
                }
            }
        }
    }

    // Select the most correct pose.
    uint32_t max_score = 0;
    for (Statis &item : statis) {
        if (item.score >= max_score) {
            R_cr = Mat3(item.q_cr);
            t_cr = item.p_cr;
            max_score = item.score;
        }
    }

    return true;
}

}
