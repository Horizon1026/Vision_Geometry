#include "point_triangulator.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

bool PointTriangulator::SolveAnalyticConstraint(const Eigen::Matrix<float, Eigen::Dynamic, 4> &A, Vec3 &p_w) {
    const Vec4 x = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    RETURN_FALSE_IF(std::fabs(x(3)) < kZeroFloat);
    p_w = x.head<3>() / x(3);
    return true;
}

}  // namespace vision_geometry
