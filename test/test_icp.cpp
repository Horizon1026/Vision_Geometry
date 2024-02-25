#include "fstream"
#include "iostream"
#include "cmath"

#include "geometry_icp.h"
#include "log_report.h"
#include "visualizor_3d.h"

using namespace SLAM_UTILITY;
using namespace SLAM_VISUALIZOR;

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Iterative-Closest Point Module Test" RESET_COLOR);
    LogFixPercision(3);

    // Set ground truth of relative pose.
    const Quat gt_q_rc = Quat(0.9, 0.05, 0.05, 0.05).normalized();
    const Vec3 gt_p_rc = Vec3::Ones() * 3.0;

    // Create two point clouds.
    std::vector<Vec3> ref_p_w;
    std::vector<Vec3> cur_p_w;
    for (uint32_t i = 1; i < 10; i++) {
        for (uint32_t j = 1; j < 10; j++) {
            for (uint32_t k = 1; k < 10; k++) {
                if (i != 1 && i != 9 && j != 1 && j != 9 && k != 1 && k != 9) {
                    continue;
                }
                const Vec3 p_w = Vec3(i, j, k);
                cur_p_w.emplace_back(p_w);
                ref_p_w.emplace_back(gt_q_rc * cur_p_w.back() + gt_p_rc);
            }
        }
    }

    // Estimate pose by icp solver.
    Quat q_rc = Quat::Identity();
    Vec3 p_rc = Vec3::Zero();
    VISION_GEOMETRY::IcpSolver icp_solver;
    icp_solver.options().kMethod = VISION_GEOMETRY::IcpSolver::IcpMethod::kPointToLine;
    icp_solver.options().kMaxValidRelativePointDistance = 5.0f;
    icp_solver.options().kMaxIteration = 1;

    ReportInfo("Ground truth q_rc " << LogQuat(gt_q_rc));
    ReportInfo("Ground truth p_rc " << LogVec(gt_p_rc));

    uint32_t cnt = 0;
    bool is_converged = false;
    Visualizor3D::camera_view().p_wc = Vec3(5, 5, -20);
    while (!Visualizor3D::ShouldQuit()) {
        ++cnt;
        if (cnt < 2) {
            Visualizor3D::Refresh("ICP [ref|RED] [cur|GREEN] [estimate|ORANGE(should be the same as cur)]", 50);
            continue;
        } else {
            cnt = 0;
        }

        if (is_converged) {
            continue;
        }

        // Iterate ICP once.
        ReportInfo("Iterate once.");
        const Quat last_q_rc = q_rc;
        const Vec3 last_p_rc = p_rc;
        icp_solver.EstimatePose(ref_p_w, cur_p_w, q_rc, p_rc);
        ReportInfo("Estimated q_rc " << LogQuat(q_rc));
        ReportInfo("Estimated p_rc " << LogVec(p_rc));
        if ((last_q_rc.inverse() * q_rc).vec().norm() + (last_p_rc - p_rc).norm() < 1e-6) {
            is_converged = true;
        }

        // Visualize.
        Visualizor3D::Clear();
        for (const auto &point : ref_p_w) {
            Visualizor3D::points().emplace_back(PointType{
                .p_w = point,
                .color = RgbColor::kRed,
                .radius = 3,
            });
        }
        for (const auto &point : cur_p_w) {
            Visualizor3D::points().emplace_back(PointType{
                .p_w = point,
                .color = RgbColor::kGreen,
                .radius = 3,
            });
        }
        for (const auto &point : cur_p_w) {
            Visualizor3D::points().emplace_back(PointType{
                .p_w = q_rc * point + p_rc,
                .color = RgbColor::kOrange,
                .radius = 2,
            });
        }


    }

    return 0;
}
