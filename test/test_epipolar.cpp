#include "cmath"
#include "fstream"
#include "iostream"

#include "geometry_epipolar.h"
#include "slam_log_reporter.h"

void TestEssentialFivePointsModel(VISION_GEOMETRY::EpipolarSolver &solver, std::vector<Vec2> &ref_norm_xy, std::vector<Vec2> &cur_norm_xy,
                                  std::vector<uint8_t> &status) {
    clock_t begin, end;
    Mat3 essential;
    Mat3 R_cr;
    Vec3 t_cr;

    ReportColorInfo(">> Test epipolar using five points model.");
    solver.options().kModel = VISION_GEOMETRY::EpipolarSolver::EpipolarModel::kFivePoints;

    begin = clock();
    solver.EstimateEssential(ref_norm_xy, cur_norm_xy, essential, status);
    end = clock();

    float cost_time = static_cast<float>(end - begin) / CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("essential is\n" << essential);

    solver.RecoverPoseFromEssential(ref_norm_xy, cur_norm_xy, essential, R_cr, t_cr);
    ReportInfo("R_cr is\n" << R_cr);
    ReportInfo("t_cr is " << t_cr.transpose());
}

void TestEssentialEightPointsModel(VISION_GEOMETRY::EpipolarSolver &solver, std::vector<Vec2> &ref_norm_xy, std::vector<Vec2> &cur_norm_xy,
                                   std::vector<uint8_t> &status) {
    clock_t begin, end;
    Mat3 essential;
    Mat3 R_cr;
    Vec3 t_cr;

    ReportColorInfo(">> Test epipolar using eight points model.");
    solver.options().kModel = VISION_GEOMETRY::EpipolarSolver::EpipolarModel::kEightPoints;

    begin = clock();
    solver.EstimateEssential(ref_norm_xy, cur_norm_xy, essential, status);
    end = clock();

    float cost_time = static_cast<float>(end - begin) / CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("essential is\n" << essential);

    solver.RecoverPoseFromEssential(ref_norm_xy, cur_norm_xy, essential, R_cr, t_cr);
    ReportInfo("R_cr is\n" << R_cr);
    ReportInfo("t_cr is " << t_cr.transpose());
}


int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test epipolar estimator." RESET_COLOR);
    LogFixPercision(8);

    // Generate 3d point cloud.
    std::vector<Vec3> points;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                points.emplace_back(Vec3(i * 5.0, j * 3.0, k * 10.0));
            }
        }
    }

    // Define two camera poses of camera view.
    Mat3 R_c0w = Mat3::Identity();
    Vec3 t_c0w = Vec3::Zero();
    Mat3 R_c1w = Mat3::Identity();
    Vec3 t_c1w = Vec3(2, -10, 6);

    // Generate observations.
    std::vector<Vec2> ref_norm_xy, cur_norm_xy;
    for (uint32_t i = 0; i < points.size(); i++) {
        Vec3 p_c = R_c0w * points[i] + t_c0w;
        ref_norm_xy.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
        p_c = R_c1w * points[i] + t_c1w;
        cur_norm_xy.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
    }

    VISION_GEOMETRY::EpipolarSolver solver;
    solver.options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::kRansac;
    std::vector<uint8_t> status;

    TestEssentialEightPointsModel(solver, ref_norm_xy, cur_norm_xy, status);
    TestEssentialFivePointsModel(solver, ref_norm_xy, cur_norm_xy, status);

    return 0;
}
