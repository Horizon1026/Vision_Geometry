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
    // TODO:

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
        const Vec3 pts0 = Vec3(norm_uv_ref[i].x(), norm_uv_ref[i].y(), 1.0f);
        const Vec3 pts1 = Vec3(norm_uv_cur[i].x(), norm_uv_cur[i].y(), 1.0f);
        residuals[i] = (pts1.transpose() * essential * pts0)(0, 0);
    }
}

}
