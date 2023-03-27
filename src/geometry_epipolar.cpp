#include "geometry_epipolar.h"
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

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                             const std::vector<Vec2> &norm_uv_cur,
                                             Mat3 &essential,
                                             std::vector<EpipolarResult> &status) {
    RETURN_FALSE_IF_FALSE(EstimateEssentialUseAll(norm_uv_ref, norm_uv_cur, essential));

    if (status.size() != norm_uv_ref.size()) {
        status.resize(norm_uv_ref.size(), EpipolarResult::SOLVED);
    }

    // Check those features that haven't been solved.
    for (uint32_t i = 0; i < norm_uv_ref.size(); ++i) {
        if (status[i] == EpipolarResult::UNSOLVED) {
            const Vec3 p_ref = Vec3(norm_uv_ref[i].x(), norm_uv_ref[i].y(), 1);
            const Vec3 p_cur = Vec3(norm_uv_cur[i].x(), norm_uv_cur[i].y(), 1);

            const float residual = p_ref.transpose() * essential * p_cur;
            if (residual < options_.kMaxEpipolarResidual) {
                status[i] = EpipolarResult::SOLVED;
            } else {
                status[i] = EpipolarResult::LARGE_RISIDUAL;
            }
        }
    }

    RefineEssentialMatrix(essential);

    Mat3 R0, R1;
    Vec3 t0, t1;
    DecomposeEssentialMatrix(essential, R0, R1, t0, t1);

    return true;
}

bool EpipolarSolver::EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                             const std::vector<Vec2> &norm_uv_cur,
                                             Mat3 &essential) {
    if (norm_uv_ref.size() != norm_uv_cur.size() || norm_uv_ref.size() < 8) {
        return false;
    }

    const uint32_t rows = std::min(options_.kMaxSolvePointsNumber, static_cast<uint32_t>(norm_uv_ref.size()));
    A.setZero(rows, 9);

    for (uint32_t i = 0; i < rows; ++i) {
        const float u1 = norm_uv_ref[i].x();
        const float v1 = norm_uv_ref[i].y();
        const float u2 = norm_uv_cur[i].x();
        const float v2 = norm_uv_cur[i].y();
        A.row(i) << u2 * v1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1;
    }

    Vec9 e = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    LogInfo("e is " << e.transpose());
    essential << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

    return true;
}

bool EpipolarSolver::EstimateEssentialRansac(const std::vector<Vec2> &norm_uv_ref,
                                             const std::vector<Vec2> &norm_uv_cur,
                                             Mat3 &essential,
                                             std::vector<EpipolarResult> &status) {

    return true;
}

void EpipolarSolver::RefineEssentialMatrix(Mat3 &essential) {
    Eigen::JacobiSVD<Mat3> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    essential = svd.matrixU() * Vec3(1, 1, 0).diagonal() * svd.matrixV();
}

void EpipolarSolver::DecomposeEssentialMatrix(const Mat3 &essential, Mat3 &R0, Mat3 &R1, Vec3 &t0, Vec3 &t1) {
    Eigen::JacobiSVD<Mat3> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();

    Mat3 R;
    R << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R0 = U * R * V.transpose();
    R1 = U * R.transpose() * V.transpose();
    t0 = U.col(2);
    t1 = - t0;

    LogInfo("R0 is\n" << R0);
    LogInfo("t0 is " << t0.transpose());
    LogInfo("R1 is\n" << R1);
    LogInfo("t1 is " << t1.transpose());
}

}
