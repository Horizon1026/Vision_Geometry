#include "fstream"
#include "iostream"

#include "point_triangulator.h"
#include "slam_log_reporter.h"

using namespace VISION_GEOMETRY;

void TestTriangulatorOnce(const std::string &method_name,
                          const PointTriangulator::Method &method,
                          const std::vector<Vec2> &observe_vec,
                          const std::vector<Quat> &q_wc_vec,
                          const std::vector<Vec3> &p_wc_vec,
                          const Vec3 &gt_p_w) {
    float cost_time;
    clock_t begin, end;
    PointTriangulator solver;
    const Vec3 p_w_noise = Vec3(0.5f, 0.5f, 0.5f);
    Vec3 res_p_w = gt_p_w + p_w_noise;

    ReportColorInfo(">> Test point_triangulator using " << method_name << " method.");
    solver.options().kMethod = method;
    begin = clock();
    solver.Triangulate(q_wc_vec, p_wc_vec, observe_vec, res_p_w);
    end = clock();
    cost_time = static_cast<float>(end - begin)/ CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time " << cost_time << " ms");
    ReportInfo("res p_w is " << res_p_w.transpose());
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test point triangulator." RESET_COLOR);
    LogFixPercision(3);

    const uint32_t number_of_camera_views = 16;      // 相机数目
    const Vec3 p_w{2, 2, 2};
    std::vector<Vec2> observe_vec;
    std::vector<Quat> q_wc_vec;
    std::vector<Vec3> p_wc_vec;

    const float radius = 8;
    for (uint32_t n = 0; n < number_of_camera_views; ++n) {
        const float theta = n * 2 * M_PI / (number_of_camera_views * 16);
        const Mat3 R_cw = Eigen::AngleAxis<float>(theta, Vec3::UnitZ()).toRotationMatrix();
        const Vec3 p_cw = Vec3(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        const Vec3 p_c = R_cw * p_w + p_cw;
        observe_vec.emplace_back(p_c[0] / p_c[2], p_c[1] / p_c[2]);
        q_wc_vec.emplace_back(R_cw.transpose());
        p_wc_vec.emplace_back(- R_cw.transpose() * p_cw);
    }

    TestTriangulatorOnce("analytic", PointTriangulator::Method::kAnalytic, observe_vec, q_wc_vec, p_wc_vec, p_w);
    TestTriangulatorOnce("optimize", PointTriangulator::Method::kOptimize, observe_vec, q_wc_vec, p_wc_vec, p_w);
    TestTriangulatorOnce("optimize huber", PointTriangulator::Method::kOptimizeHuber, observe_vec, q_wc_vec, p_wc_vec, p_w);
    TestTriangulatorOnce("optimize cauchy", PointTriangulator::Method::kOptimizeCauchy, observe_vec, q_wc_vec, p_wc_vec, p_w);

    ReportColorInfo(">> Test computation of parallex angle.");
    PointTriangulator solver;
    for (uint32_t i = 1; i < q_wc_vec.size(); ++i) {
        const float parallex_angle = solver.GetSineOfParallexAngle(q_wc_vec[0], p_wc_vec[0], q_wc_vec[i], p_wc_vec[i], observe_vec[0], observe_vec[i]);
        ReportInfo("Feature " << LogVec(p_w) << " has sine parallex angle [" << parallex_angle << "] between camera pose 0 and pose " << i << "");
    }

    return 0;
}
