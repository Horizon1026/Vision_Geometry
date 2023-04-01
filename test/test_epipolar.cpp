/* 外部依赖 */
#include <fstream>
#include <iostream>
#include "cmath"

/* 内部依赖 */
#include "geometry_epipolar.h"
#include "log_api.h"

void TestEssentialFivePointsModel(VISION_GEOMETRY::EpipolarSolver &solver,
                                  std::vector<Vec2> &norm_uv_ref,
                                  std::vector<Vec2> &norm_uv_cur,
                                  std::vector<VISION_GEOMETRY::EpipolarSolver::EpipolarResult> &status) {
    clock_t begin, end;
    Mat3 essential;
    Mat3 R_cr;
    Vec3 t_cr;

    LogInfo(GREEN ">> Test epipolar using eight points model." RESET_COLOR);
    solver.options().kModel = VISION_GEOMETRY::EpipolarSolver::EpipolarModel::FIVE_POINTS;

    begin = clock();
    solver.EstimateEssential(norm_uv_ref, norm_uv_cur, essential, status);
    end = clock();

    float cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("essential is\n" << essential);

    solver.RecoverPoseFromEssential(norm_uv_ref, norm_uv_cur, essential, R_cr, t_cr);
    LogInfo("R_cr is\n" << R_cr);
    LogInfo("t_cr is " << t_cr.transpose());
}

void TestEssentialEightPointsModel(VISION_GEOMETRY::EpipolarSolver &solver,
                                   std::vector<Vec2> &norm_uv_ref,
                                   std::vector<Vec2> &norm_uv_cur,
                                   std::vector<VISION_GEOMETRY::EpipolarSolver::EpipolarResult> &status) {
    clock_t begin, end;
    Mat3 essential;
    Mat3 R_cr;
    Vec3 t_cr;

    LogInfo(GREEN ">> Test epipolar using eight points model." RESET_COLOR);
    solver.options().kModel = VISION_GEOMETRY::EpipolarSolver::EpipolarModel::EIGHT_POINTS;

    begin = clock();
    solver.EstimateEssential(norm_uv_ref, norm_uv_cur, essential, status);
    end = clock();

    float cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("essential is\n" << essential);

    solver.RecoverPoseFromEssential(norm_uv_ref, norm_uv_cur, essential, R_cr, t_cr);
    LogInfo("R_cr is\n" << R_cr);
    LogInfo("t_cr is " << t_cr.transpose());
}


int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Epipolar Module Test" RESET_COLOR);
    LogFixPercision(8);

    // 构造 3D 点云
    std::vector<Vec3> points;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                points.emplace_back(Vec3(i * 5.0, j * 3.0, k * 10.0));
            }
        }
    }

    // 定义两帧位姿
    Mat3 R_c0w = Mat3::Identity();
    Vec3 t_c0w = Vec3::Zero();
    Mat3 R_c1w = Mat3::Identity();
    Vec3 t_c1w = Vec3(2, -10, 0);

    // 将 3D 点云通过两帧位姿映射到对应的归一化平面上，构造匹配点对
    std::vector<Vec2> norm_uv_ref, norm_uv_cur;
    for (uint32_t i = 0; i < points.size(); i++) {
        Vec3 p_c = R_c0w * points[i] + t_c0w;
        norm_uv_ref.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
        p_c = R_c1w * points[i] + t_c1w;
        norm_uv_cur.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
    }

    VISION_GEOMETRY::EpipolarSolver solver;
    solver.options().kMethod = VISION_GEOMETRY::EpipolarSolver::EpipolarMethod::EPIPOLAR_RANSAC;
    std::vector<VISION_GEOMETRY::EpipolarSolver::EpipolarResult> status;

    TestEssentialEightPointsModel(solver, norm_uv_ref, norm_uv_cur, status);

    TestEssentialFivePointsModel(solver, norm_uv_ref, norm_uv_cur, status);

    return 0;
}
