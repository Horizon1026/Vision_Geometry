/* 外部依赖 */
#include <fstream>
#include <iostream>
#include "cmath"

/* 内部依赖 */
#include "geometry_pnp.h"
#include "log_api.h"


int main(int argc, char **argv) {
    LogInfo(YELLOW ">> Perspective-n-Point Module Test" RESET_COLOR);
    LogFixPercision();

    // 构造 3D 点云
    std::vector<Vec3> pts_3d;
    for (uint32_t i = 1; i < 10; i++) {
        for (uint32_t j = 1; j < 10; j++) {
            for (uint32_t k = 1; k < 10; k++) {
                pts_3d.emplace_back(Vec3(i, j, k * 2.0));
            }
        }
    }

    // 定义相机位姿（相对于世界坐标系）
    Mat3 R_wc = Mat3::Identity();
    Vec3 p_wc = Vec3(1, -3, 0);
    Quat q_wc(R_wc);
    LogInfo("true q_wc is " << LogQuat(q_wc));
    LogInfo("true p_wc is " << LogVec(p_wc));

    // 将 3D 点云通过两帧位姿映射到对应的归一化平面上，构造匹配点对
    std::vector<Vec2> pts_2d;
    for (uint32_t i = 0; i < pts_3d.size(); i++) {
        Vec3 p_c = R_wc.transpose() * (pts_3d[i] - p_wc);
        pts_2d.emplace_back(Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)));
    }

    // 随机给匹配点增加 outliers
    std::vector<int> outliers_indice;
    for (uint32_t i = 0; i < pts_3d.size() / 100; i++) {
        uint32_t idx = rand() % pts_3d.size();
        pts_2d[idx](0, 0) = 10;
        outliers_indice.emplace_back(idx);
    }

    Quat res_q_wc;
    Vec3 res_p_wc;
    VISION_GEOMETRY::PnpSolver pnpSolver;
    std::vector<VISION_GEOMETRY::PnpSolver::PnpResult> status;
    float cost_time;
    clock_t begin, end;

    LogInfo(GREEN ">> Test pnp using all points." RESET_COLOR);
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PNP_ALL;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    LogInfo(GREEN ">> Test pnp using ransac method." RESET_COLOR);
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PNP_RANSAC;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    LogInfo(GREEN ">> Test pnp using huber kernel." RESET_COLOR);
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PNP_HUBER;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    LogInfo(GREEN ">> Test pnp using cauchy kernel." RESET_COLOR);
    res_q_wc.setIdentity();
    res_p_wc.setZero();
    pnpSolver.options().kMethod = VISION_GEOMETRY::PnpSolver::PNP_CAUCHY;
    begin = clock();
    pnpSolver.EstimatePose(pts_3d, pts_2d, res_q_wc, res_p_wc, status);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    LogInfo("cost time is " << cost_time << " ms");
    LogInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc));

    return 0;
}