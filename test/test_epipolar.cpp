/* 外部依赖 */
#include <fstream>
#include <iostream>
#include "cmath"

/* 内部依赖 */
#include "geometry_epipolar.h"
#include "log_api.h"


int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Epipolar Module Test" RESET_COLOR);
    LogFixPercision(3);

    // 构造 3D 点云
    std::vector<Vec3> points;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                points.emplace_back(Vec3(i, j, k * 2.0));
            }
        }
    }

    // 定义两帧位姿
    Mat3 R_c0w = Mat3::Identity();
    Vec3 t_c0w = Vec3::Zero();
    Mat3 R_c1w = Mat3::Identity();
    Vec3 t_c1w = Vec3(1, -2, 0);

    // 将 3D 点云通过两帧位姿映射到对应的归一化平面上，构造匹配点对
    std::vector<Vec2> norm_uv_ref, norm_uv_cur;
    for (uint32_t i = 0; i < points.size(); i++) {
        Vec3 p_c = R_c0w * points[i] + t_c0w;
        norm_uv_ref.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
        p_c = R_c1w * points[i] + t_c1w;
        norm_uv_cur.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
    }

    VISION_GEOMETRY::EpipolarSolver solver;
    float cost_time;
    clock_t begin, end;

    LogInfo(GREEN ">> Test epipolar using all points." RESET_COLOR);
    Mat3 essential;
    std::vector<VISION_GEOMETRY::EpipolarSolver::EpipolarResult> status;
    begin = clock();
    solver.EstimateEssential(norm_uv_ref, norm_uv_cur, essential, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("essential is\n" << essential);

    LogInfo(GREEN ">> Decompose essential matrix." RESET_COLOR);
    Mat3 R0, R1;
    Vec3 t0, t1;
    solver.DecomposeEssentialMatrix(essential, R0, R1, t0, t1);
    LogInfo("R0 is\n" << R0);
    LogInfo("t0 is " << t0.transpose());

    return 0;
}
