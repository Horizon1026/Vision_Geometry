#include "fstream"
#include "iostream"
#include "cmath"

#include "geometry_icp.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "visualizor_3d.h"

using namespace SLAM_UTILITY;
using namespace SLAM_VISUALIZOR;

void LoadLidarScan(const std::string &file_name, const Vec3 &offset, std::vector<Vec3> &point_cloud) {
    std::ifstream file;
    file.open(file_name.c_str());
    if (!file.is_open()) {
        ReportError("Failed to open file: " + file_name);
        return;
    }
    point_cloud.clear();
    point_cloud.reserve(30000);

    std::string one_line;
    Vec3 pos = Vec3::Zero();
    while (std::getline(file, one_line) && !one_line.empty()) {
        std::istringstream imu_data(one_line);
        imu_data >> pos.x() >> pos.y() >> pos.z();
        point_cloud.emplace_back(pos + offset);
    }
    file.close();
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test iterative-closest point." RESET_COLOR);
    LogFixPercision(3);

    // Create two point clouds.
    std::vector<Vec3> ref_p_w;
    std::vector<Vec3> cur_p_w;
    LoadLidarScan("../examples/cur_lidar_scan.txt", Vec3::Zero(), cur_p_w);
    LoadLidarScan("../examples/ref_lidar_scan.txt", Vec3(5, 2, 1), ref_p_w);

    // Estimate pose by icp solver.
    Quat q_rc = Quat::Identity();
    Vec3 p_rc = Vec3::Zero();
    VISION_GEOMETRY::IcpSolver icp_solver;
    icp_solver.options().kMethod = VISION_GEOMETRY::IcpSolver::IcpMethod::kPointToPlane;
    icp_solver.options().kUseNanoFlannKdTree = true;
    icp_solver.options().kMaxValidRelativePointDistance = 5.0f;
    icp_solver.options().kMaxIteration = 1;

    uint32_t cnt = 0;
    bool is_converged = false;
    Visualizor3D::camera_view().q_wc = Quat(0.3f, -0.9f, 0.0f, 0.0f).normalized();
    Visualizor3D::camera_view().p_wc = Vec3(5, -25, 25);
    while (!Visualizor3D::ShouldQuit()) {
        ++cnt;
        if (cnt < 5) {
            Visualizor3D::Refresh("ICP [ ref | RED ] [ cur | GREEN ] [ estimate | ORANGE(should be the same as ref) ]", 50);
            continue;
        } else {
            cnt = 0;
        }

        CONTINUE_IF(is_converged);

        // Iterate ICP once.
        ReportInfo("Iterate once.");
        const Quat last_q_rc = q_rc;
        const Vec3 last_p_rc = p_rc;
        icp_solver.EstimatePose(ref_p_w, cur_p_w, q_rc, p_rc);
        ReportInfo("Estimated q_rc " << LogQuat(q_rc));
        ReportInfo("Estimated p_rc " << LogVec(p_rc));
        if ((last_q_rc.inverse() * q_rc).vec().norm() + (last_p_rc - p_rc).norm() < 1e-4) {
            is_converged = true;
        }

        // Visualize.
        Visualizor3D::Clear();
        for (const auto &point: ref_p_w) {
            Visualizor3D::points().emplace_back(PointType{
                .p_w = point,
                .color = RgbColor::kRed,
                .radius = 1,
            });
        }
        for (const auto &point: cur_p_w) {
            Visualizor3D::points().emplace_back(PointType{
                .p_w = point,
                .color = RgbColor::kGreen,
                .radius = 1,
            });
        }
        for (const auto &point: cur_p_w) {
            Visualizor3D::points().emplace_back(PointType{
                .p_w = q_rc * point + p_rc,
                .color = RgbColor::kOrange,
                .radius = 2,
            });
        }
    }

    return 0;
}
