#include "cmath"
#include "fstream"
#include "iostream"

#include "relative_rotation.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"


int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test pure relative rotation estimator." RESET_COLOR);
    LogFixPercision(8);

    // Create 3d point cloud.
    std::vector<Vec3> points;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                points.emplace_back(Vec3(i * 5.0, j * 3.0, k * 10.0));
            }
        }
    }

    // Create two camera frame with pose in word frame.
    Mat3 R_rw = Mat3::Identity();
    Vec3 t_rw = Vec3::Zero();
    Mat3 R_cw = Quat(1.0, 0.4, 0, 0).normalized().matrix();
    Vec3 t_cw = Vec3(1, 1, 1);

    // Compute pairs of features in two camera frames.
    std::vector<Vec2> ref_norm_xy, cur_norm_xy;
    for (uint32_t i = 0; i < points.size(); ++i) {
        const Vec3 p_r = R_rw * points[i] + t_rw;
        ref_norm_xy.emplace_back(Vec2(p_r(0) / p_r(2), p_r(1) / p_r(2)));
        const Vec3 p_c = R_cw * points[i] + t_cw;
        cur_norm_xy.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
    }

    vision_geometry::RelativeRotation solver;
    Quat q_cr = Quat::Identity();
    Vec3 t_cr = Vec3::Zero();
    Vec3 euler_rpy = Vec3::Zero();

    // Show the ground truth.
    q_cr = Quat(R_cw * R_rw.transpose());
    euler_rpy = Utility::QuaternionToEuler(q_cr);
    ReportInfo("Ground truh q_cr is " << LogQuat(q_cr) << ", euler_rpy(deg) is " << LogVec(euler_rpy));
    ReportInfo("Ground truh t_cr is " << LogVec(t_cw));

    // Test estimate both rotation and translation.
    q_cr.setIdentity();
    solver.EstimatePose(ref_norm_xy, cur_norm_xy, q_cr, t_cr);
    euler_rpy = Utility::QuaternionToEuler(q_cr);
    ReportInfo("Estimated q_cr is " << LogQuat(q_cr) << ", euler_rpy(deg) is " << LogVec(euler_rpy));
    ReportInfo("Estimated t_cr is " << LogVec(t_cr));

    // Test only estimate rotation.
    q_cr.setIdentity();
    solver.EstimateRotation(ref_norm_xy, cur_norm_xy, q_cr);
    euler_rpy = Utility::QuaternionToEuler(q_cr);
    ReportInfo("Directly estimated q_cr is " << LogQuat(q_cr) << ", euler_rpy(deg) is " << LogVec(euler_rpy));

    // Test only estimate rotation with bnb.
    q_cr.setIdentity();
    solver.EstimateRotationByBnb(ref_norm_xy, cur_norm_xy, q_cr);
    euler_rpy = Utility::QuaternionToEuler(q_cr);
    ReportInfo("Bnb estimated q_cr is " << LogQuat(q_cr) << ", euler_rpy(deg) is " << LogVec(euler_rpy));

    return 0;
}
