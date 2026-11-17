#include "point_triangulator.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

bool PointTriangulator::IncrementalTriangulate(const Vec3 &p_wc, const Quat &q_wc, const Vec2 &norm_xy, Vec3 &p_w) {
    return IncrementalTriangulate(p_wc, q_wc, Vec3(norm_xy.x(), norm_xy.y(), 1.0f).normalized(), p_w);
}

bool PointTriangulator::IncrementalTriangulate(const Vec3 &p_wc, const Quat &q_wc, const Vec3 &bearing_xyz, Vec3 &p_w) {
    RETURN_FALSE_IF(!p_wc.allFinite() || !q_wc.coeffs().allFinite() || !bearing_xyz.allFinite());
    RETURN_FALSE_IF(bearing_xyz.norm() <= kZeroFloat);
    UpdateIncrementalFunction(p_wc, q_wc, bearing_xyz);

    Eigen::ColPivHouseholderQR<Mat3> solver(A_);
    solver.setThreshold(kZeroFloat);
    RETURN_FALSE_IF(solver.rank() < 3);
    const Vec3 result = solver.solve(b_);
    RETURN_FALSE_IF(!result.allFinite());
    p_w = result;
    return true;
}

void PointTriangulator::Reset() {
    A_.setZero();
    b_.setZero();
}

void PointTriangulator::UpdateIncrementalFunction(const Vec3 &p_wc, const Quat &q_wc, const Vec3 &bearing_xyz) {
    // Define the unit ray and camera center in the world frame:
    //   d = R_wc * bearing_xyz / ||bearing_xyz||
    //   c = p_wc
    // Let p be the unknown world point. The ray constraint is:
    //   [d]_x * (p - c) = 0
    //
    // Minimize the sum of squared distances to the observation rays:
    //   min_p sum_i ||[d_i]_x * (p - c_i)||^2
    // The resulting normal equation is:
    //   A * p = b
    //   A = sum_i [d_i]_x^T * [d_i]_x
    //   b = sum_i [d_i]_x^T * [d_i]_x * c_i
    //
    // Since ||d_i|| = 1:
    //   [d_i]_x^T * [d_i]_x = I - d_i * d_i^T
    // Each new observation contributes:
    //   A_i = I - d_i * d_i^T
    //   b_i = A_i * c_i
    // Update the accumulated system:
    //   A <- A + A_i
    //   b <- b + b_i
    const Vec3 bearing_w = (q_wc * bearing_xyz).normalized();
    const Mat3 A_i = Mat3::Identity() - bearing_w * bearing_w.transpose();
    A_ += A_i;
    b_ += A_i * p_wc;
}

float PointTriangulator::Huber(float param, float x) {
    float huber = 1.0f;
    if (x > param) {
        huber = param / x;
    }
    return huber;
}

float PointTriangulator::Cauchy(float param, float x) {
    const float param2 = param * param;
    return param2 / (param2 + x * x);
}

}  // namespace vision_geometry
