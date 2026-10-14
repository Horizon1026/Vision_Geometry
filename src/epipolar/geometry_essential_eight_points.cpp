#include "geometry_epipolar.h"

#include "slam_operations.h"

#include "limits"

namespace vision_geometry {

bool EpipolarSolver::SolveEightPointConstraint(const Mat &A, Mat3 &essential, bool check_rank) {
    Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
    if (check_rank) {
        RETURN_FALSE_IF(svd.singularValues()(7) <= std::numeric_limits<float>::epsilon() * svd.singularValues()(0));
    }
    const Vec9 e = svd.matrixV().rightCols<1>();
    essential << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);
    return true;
}

}  // namespace vision_geometry
