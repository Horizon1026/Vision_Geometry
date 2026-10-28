#include "geometry_homography_pnp.h"

#include "slam_log_reporter.h"

#include "cmath"
#include "fstream"
#include "iostream"

using namespace vision_geometry;

void TestPnpOnce(const std::string method_name, const HomographyPnpSolver::Method method, const std::vector<Vec3> &ref_bearing_xyz, const float &depth_m,
                 const std::vector<Vec3> &cur_bearing_xyz) {
    Quat res_q_cw = Quat::Identity();
    Vec3 res_p_cw = Vec3::Zero();
    Vec2 homo_plane_slope_xy = Vec2::Zero();
    std::vector<uint8_t> status;
    float cost_time = 0.0f;
    clock_t begin = 0;
    clock_t end = 0;

    HomographyPnpSolver solver {};
    solver.options().kMethod = method;

    ReportColorInfo(">> Test pnp with [bearing_xyz] " << method_name << ".");
    begin = clock();
    solver.EstimatePose(ref_bearing_xyz, depth_m, cur_bearing_xyz, res_p_cw, res_q_cw, homo_plane_slope_xy, status);
    end = clock();
    cost_time = static_cast<float>(end - begin) / CLOCKS_PER_SEC * 1000.0f;
    ReportInfo("cost time is " << cost_time << " ms");

    const Vec3 res_p_wc = -(res_q_cw.inverse() * res_p_cw);
    const Quat res_q_wc = res_q_cw.inverse();
    ReportInfo("res_q_wc is " << LogQuat(res_q_wc) << ", res_p_wc is " << LogVec(res_p_wc) << ", homo_plane_slope_xy is " << LogVec(homo_plane_slope_xy));
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test perspective-n-point." RESET_COLOR);
    LogFixPercision(3);

    // Generate 3d point cloud.
    std::vector<Vec3> pts_3d;
    const float depth_m = 1.0f;
    const Vec2 slope_xy = Vec2(0.02f, 0.03f);
    for (uint32_t i = 1; i < 10; i++) {
        for (uint32_t j = 1; j < 10; j++) {
            pts_3d.emplace_back(Vec3(i, j, depth_m - slope_xy.x() * i - slope_xy.y() * j));
        }
    }

    // Define camera pose of camera view.
    Mat3 R_wc = Mat3::Identity();
    Vec3 p_wc = Vec3(1, -3, 0);
    Quat q_wc(R_wc);
    ReportInfo("true p_wc is " << LogVec(p_wc) << ", true q_wc is " << LogQuat(q_wc) << ", true slope xy is " << LogVec(slope_xy));

    // Generate observations.
    std::vector<Vec3> ref_bearing_xyz;
    std::vector<Vec3> cur_bearing_xyz;
    for (uint32_t i = 0; i < pts_3d.size(); i++) {
        ref_bearing_xyz.emplace_back(pts_3d[i].normalized());
        const Vec3 p_c = R_wc.transpose() * (pts_3d[i] - p_wc);
        cur_bearing_xyz.emplace_back(p_c.normalized());
    }

    // Add outliers.
    // std::vector<int> outliers_indice;
    // for (uint32_t i = 0; i < pts_3d.size() / 100; i++) {
    //     uint32_t idx = rand() % pts_3d.size();
    //     ref_bearing_xyz[idx].x() += 0.1f * ref_bearing_xyz[idx].z();
    //     ref_bearing_xyz[idx].normalize();
    //     cur_bearing_xyz[idx].x() += 0.1f * cur_bearing_xyz[idx].z();
    //     cur_bearing_xyz[idx].normalize();
    //     outliers_indice.emplace_back(idx);
    // }

    TestPnpOnce("using all points by optimization", HomographyPnpSolver::Method::kOptimize, ref_bearing_xyz, depth_m, cur_bearing_xyz);

    return 0;
}
