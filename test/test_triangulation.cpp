#include "fstream"
#include "iostream"

#include "geometry_triangulation.h"
#include "log_report.h"

VISION_GEOMETRY::Triangulator solver;

bool TestTrianglateAnalytic() {
    uint32_t poseNums = 6;      // 相机数目

    Vec3 p_w{2, 2, 2};
    std::vector<Vec2> observe_vec;
    std::vector<Quat> q_wc_vec;
    std::vector<Vec3> p_wc_vec;

    float radius = 8;
    for (uint32_t n = 0; n < poseNums; ++n) {
        float theta = n * 2 * M_PI / (poseNums * 16); // 1/16 圆弧
        // 绕 z 轴 旋转
        Mat3 R_cw;
        R_cw = Eigen::AngleAxis<float>(theta, Vec3::UnitZ());
        Vec3 p_cw = Vec3(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        //cameraPoses.push_back(Frame(R_cw, p_cw));
        auto p_c = R_cw * p_w + p_cw;
        observe_vec.emplace_back(p_c[0] / p_c[2], p_c[1] / p_c[2]);
        q_wc_vec.emplace_back(R_cw.transpose());
        p_wc_vec.emplace_back(- R_cw.transpose() * p_cw);
    }

    Vec3 res_p_w;
    solver.options().kMethod = VISION_GEOMETRY::Triangulator::TriangulationMethod::kAnalytic;
    solver.Triangulate(q_wc_vec, p_wc_vec, observe_vec, res_p_w);
    std::cout << "TestTrianglateAnalytic :";
    std::cout << "set p_w is " << p_w.transpose() << ", res p_w is " << res_p_w.transpose() << std::endl;
    return true;
}


bool TestTrianglateIterative() {
    uint32_t poseNums = 6;      // 相机数目
    Vec3 p_w{2, 2, 2};
    std::vector<Vec2> observe_vec;
    std::vector<Quat> q_wc_vec;
    std::vector<Vec3> p_wc_vec;

    float radius = 8;
    for (uint32_t n = 0; n < poseNums; ++n) {
        float theta = n * 2 * M_PI / (poseNums * 16); // 1/16 圆弧
        // 绕 z 轴 旋转
        Mat3 R_cw;
        R_cw = Eigen::AngleAxis<float>(theta, Vec3::UnitZ());
        Vec3 p_cw = Vec3(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        auto p_c = R_cw * p_w + p_cw;
        observe_vec.emplace_back(p_c[0] / p_c[2], p_c[1] / p_c[2]);
        q_wc_vec.emplace_back(R_cw.transpose());
        p_wc_vec.emplace_back(- R_cw.transpose() * p_cw);
    }

    Vec3 p_w_noise = Vec3(0.4, 0.4, 0.4);
    Vec3 res_p_w = p_w + p_w_noise;

    solver.options().kMethod = VISION_GEOMETRY::Triangulator::TriangulationMethod::kIterative;
    solver.Triangulate(q_wc_vec, p_wc_vec, observe_vec, res_p_w);
    std::cout << "TestTrianglateIterative :";
    std::cout << "set p_w is " << res_p_w.transpose() << ", res p_w is " << res_p_w.transpose() << std::endl;
    return true;
}


int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Triangulation Module Test" RESET_COLOR);
    LogFixPercision(3);

    uint32_t poseNums = 16;      // 相机数目
    Vec3 p_w{2, 2, 2};
    std::vector<Vec2> observe_vec;
    std::vector<Quat> q_wc_vec;
    std::vector<Vec3> p_wc_vec;

    float radius = 8;
    for (uint32_t n = 0; n < poseNums; ++n) {
        float theta = n * 2 * M_PI / (poseNums * 16); // 1/16 圆弧
        // 绕 z 轴 旋转
        Mat3 R_cw;
        R_cw = Eigen::AngleAxis<float>(theta, Vec3::UnitZ());
        Vec3 p_cw = Vec3(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        //cameraPoses.push_back(Frame(R_cw, p_cw));
        auto p_c = R_cw * p_w + p_cw;
        observe_vec.emplace_back(p_c[0] / p_c[2], p_c[1] / p_c[2]);
        q_wc_vec.emplace_back(R_cw.transpose());
        p_wc_vec.emplace_back(- R_cw.transpose() * p_cw);
    }

    float cost_time;
    clock_t begin, end;

    ReportInfo(GREEN ">> Test triangulation using analytic method." RESET_COLOR);
    Vec3 res_p_w;
    solver.options().kMethod = VISION_GEOMETRY::Triangulator::TriangulationMethod::kAnalytic;
    begin = clock();
    solver.Triangulate(q_wc_vec, p_wc_vec, observe_vec, res_p_w);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("set p_w is " << p_w.transpose() << ", res p_w is " << res_p_w.transpose());

    ReportInfo(GREEN ">> Test triangulation using iterative method." RESET_COLOR);
    Vec3 p_w_noise = Vec3(0.5f, 0.5f, 0.5f);
    res_p_w = p_w + p_w_noise;;
    solver.options().kMethod = VISION_GEOMETRY::Triangulator::TriangulationMethod::kIterative;
    begin = clock();
    solver.Triangulate(q_wc_vec, p_wc_vec, observe_vec, res_p_w);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");
    ReportInfo("set p_w is " << p_w.transpose() << ", res p_w is " << res_p_w.transpose());

    return 0;
}