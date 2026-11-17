#include "point_triangulator.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

bool PointTriangulator::SolveAnalyticConstraint(const Eigen::Matrix<float, Eigen::Dynamic, 4> &A, Vec3 &p_w) {
    RETURN_FALSE_IF(A.rows() < 4 || !A.allFinite());
    const auto svd = A.jacobiSvd(Eigen::ComputeFullV);

    // Check conditioning of the three constrained directions instead: sigma_1 / sigma_3.
    const Vec4 singular_values = svd.singularValues();
    RETURN_FALSE_IF(singular_values(2) <= 0.0f);
    const float condition_number = singular_values(0) / singular_values(2);
    RETURN_FALSE_IF(!std::isfinite(condition_number) || condition_number > options_.kMaxConditionNumber);

    const Vec4 x = svd.matrixV().rightCols<1>();
    RETURN_FALSE_IF(std::fabs(x(3)) < kZeroFloat);
    p_w = x.head<3>() / x(3);
    return true;
}

}  // namespace vision_geometry
