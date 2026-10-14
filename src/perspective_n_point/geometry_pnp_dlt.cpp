#include "geometry_pnp.h"

#include "slam_operations.h"

#include "limits"

namespace vision_geometry {

bool PnpSolver::SolveDltConstraint(const Mat &A, Mat3x4 &P, bool check_rank) {
    Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
    if (check_rank) {
        RETURN_FALSE_IF(svd.singularValues()(10) <= std::numeric_limits<float>::epsilon() * svd.singularValues()(0));
    }
    const Vec v = svd.matrixV().col(11);
    P << v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11);
    return true;
}

}  // namespace vision_geometry
