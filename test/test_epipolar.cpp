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
    std::vector<Eigen::Vector3d> points;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                points.emplace_back(Eigen::Vector3d(i, j, k * 2.0));
            }
        }
    }

    // 定义两帧位姿
    Eigen::Matrix3d R_c0w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_c0w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_c1w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_c1w = Eigen::Vector3d(1, -3, 0);

    // 将 3D 点云通过两帧位姿映射到对应的归一化平面上，构造匹配点对
    std::vector<Vec2> norm_uv_ref, norm_uv_cur;
    for (unsigned long i = 0; i < points.size(); i++) {
        Eigen::Vector3d tempP = R_c0w * points[i] + t_c0w;
        norm_uv_ref.emplace_back(Vec2(tempP(0, 0) / tempP(2, 0), tempP(1, 0) / tempP(2, 0)));
        tempP = R_c1w * points[i] + t_c1w;
        norm_uv_cur.emplace_back(Vec2(tempP(0, 0) / tempP(2, 0), tempP(1, 0) / tempP(2, 0)));
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

    return 0;
}
