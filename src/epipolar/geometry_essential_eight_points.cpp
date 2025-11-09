#include "geometry_epipolar.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

#include <set>

namespace VISION_GEOMETRY {

bool EpipolarSolver::EstimateEssentialUseEightPoints(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential) {
    if (ref_norm_xy.size() != cur_norm_xy.size() || ref_norm_xy.size() < 8) {
        return false;
    }

    const uint32_t rows = std::min(options_.kMaxSolvePointsNumber, static_cast<uint32_t>(ref_norm_xy.size()));
    A.setZero(rows, 9);

    // Construct eight point model Ae = 0.
    for (uint32_t i = 0; i < rows; ++i) {
        const float u1 = ref_norm_xy[i].x();
        const float v1 = ref_norm_xy[i].y();
        const float u2 = cur_norm_xy[i].x();
        const float v2 = cur_norm_xy[i].y();
        A.row(i) << u2 * v1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1;
    }

    // Solve Ae = 0. Convert vector e to matrix E.
    Vec9 e = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    essential << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

    return true;
}

void EpipolarSolver::RefineEssentialMatrix(Mat3 &essential) {
    Eigen::JacobiSVD<Mat3> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    essential = svd.matrixU() * Vec3(1, 1, 0).asDiagonal() * svd.matrixV();
}

}  // namespace VISION_GEOMETRY
