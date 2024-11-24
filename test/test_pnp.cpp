#include "fstream"
#include "iostream"
#include "cmath"

#include "geometry_pnp.h"
#include "slam_log_reporter.h"


int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test perspective-n-point." RESET_COLOR);
    LogFixPercision(3);

    // Generate 3d point cloud.
    std::vector<Vec3> pts_3d;
    for (uint32_t i = 1; i < 10; i++) {
        for (uint32_t j = 1; j < 10; j++) {
            for (uint32_t k = 1; k < 10; k++) {
                pts_3d.emplace_back(Vec3(i, j, k * 2.0));
            }
        }
    }

    // Define camera pose of camera view.
    Mat3 R_wc = Mat3::Identity();
    Vec3 p_wc = Vec3(1, -3, 0);
    Quat q_wc(R_wc);
    ReportInfo("true q_wc is " << LogQuat(q_wc));
    ReportInfo("true p_wc is " << LogVec(p_wc));

    // Generate observations.
    std::vector<Vec2> pts_2d;
    for (uint32_t i = 0; i < pts_3d.size(); i++) {
        Vec3 p_c = R_wc.transpose() * (pts_3d[i] - p_wc);
        pts_2d.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
    }

    // Add outliers.
    std::vector<int> outliers_indice;
    for (uint32_t i = 0; i < pts_3d.size() / 100; i++) {
        uint32_t idx = rand() % pts_3d.size();
        pts_2d[idx](0, 0) = 10;
        outliers_indice.emplace_back(idx);
    }

    Quat res_q_wc;
    Vec3 res_p_wc;
    VISION_GEOMETRY::PnpSolver pnpSolver;
    std::vector<uint8_t> status;
    float cost_time;
    clock_t begin, end;

    ReportColorInfo(">> Test pnp using all points.");
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PnpMethod::kUseAll;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    ReportColorInfo(">> Test pnp using ransac method.");
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PnpMethod::kRansac;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    ReportColorInfo(">> Test pnp using huber kernel.");
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PnpMethod::kHuber;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    ReportColorInfo(">> Test pnp using cauchy kernel.");
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PnpMethod::kCauchy;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    return 0;
}