#include "fstream"
#include "iostream"

#include "line_segment.h"
#include "line_triangulator.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"

#include "visualizor_3d.h"

using namespace slam_utility;
using namespace slam_visualizor;
using namespace vision_geometry;

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test line triangulator." RESET_COLOR);
    LogFixPercision(3);

    // Generate several camera views and one line.
    const uint32_t number_of_camera_views = 5;
    const Vec3 p1_w {0, -2, 5};
    const Vec3 p2_w {0, 2, 8};
    const LinePlucker3D truth_line_w(LineSegment3D(p1_w, p2_w));
    std::vector<LineSegment2D> observe_vec;
    std::vector<Quat> q_wc_vec;
    std::vector<Vec3> p_wc_vec;

    const float radius = 8.0f;
    for (uint32_t n = 0; n < number_of_camera_views; ++n) {
        const float theta = n * 2 * M_PI / (number_of_camera_views * 16);
        const Mat3 R_wc(Eigen::AngleAxis<float>(theta, Vec3::UnitX()));
        const Vec3 p_wc = Vec3(radius * std::cos(theta) - radius + n * 1.0f, radius * std::sin(theta), 1.0f * std::sin(2 * theta));
        const Vec3 p1_c = R_wc.transpose() * (p1_w - p_wc);
        const Vec3 p2_c = R_wc.transpose() * (p2_w - p_wc);

        observe_vec.emplace_back(LineSegment2D(p1_c.head<2>() / p1_c.z(), p2_c.head<2>() / p2_c.z()));
        q_wc_vec.emplace_back(R_wc);
        p_wc_vec.emplace_back(p_wc);
    }

    // Triangulate line.
    LineTriangulator solver;
    solver.options().kMethod = LineTriangulator::Method::kOptimize;
    solver.options().kMaxIteration = 5;
    const LinePlucker3D noised_line_w(LineSegment3D(p1_w + Vec3::Random(), p2_w + Vec3::Random()));
    LinePlucker3D estimated_line_w(noised_line_w);
    ReportInfo("Initialized line in plucker is " << LogVec(estimated_line_w.param()));
    solver.Triangulate(q_wc_vec, p_wc_vec, observe_vec, estimated_line_w);

    // Report result.
    ReportInfo("Estimated line in plucker is " << LogVec(estimated_line_w.param()));
    ReportInfo("Truth line in plucker is " << LogVec(truth_line_w.param()));

    // Visualize.
    Visualizor3D::camera_view().p_wc = Vec3(0.26f, -0.65f, -4.39f);
    Visualizor3D::camera_view().q_wc = Quat(0.79f, 0.0f, 0.0f, 0.59f).normalized();
    Visualizor3D::Clear();
    Visualizor3D::poses().emplace_back(PoseType {
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 1.0f,
    });
    for (uint32_t i = 0; i < q_wc_vec.size(); ++i) {
        Visualizor3D::camera_poses().emplace_back(CameraPoseType {
            .p_wc = p_wc_vec[i],
            .q_wc = q_wc_vec[i],
            .scale = 0.5f,
        });
        // Draw observation of line.
        Visualizor3D::lines().emplace_back(LineType {
            .p_w_i = q_wc_vec[i] * observe_vec[i].start_point_homogeneous() + p_wc_vec[i],
            .p_w_j = q_wc_vec[i] * observe_vec[i].end_point_homogeneous() + p_wc_vec[i],
            .color = RgbColor::kYellow,
        });
        Visualizor3D::lines().emplace_back(LineType {
            .p_w_i = p_wc_vec[i],
            .p_w_j = p1_w,
            .color = RgbColor::kSlateGray,
        });
        Visualizor3D::lines().emplace_back(LineType {
            .p_w_i = p_wc_vec[i],
            .p_w_j = p2_w,
            .color = RgbColor::kSlateGray,
        });
    }
    // Draw groud truth line in world frame.
    Visualizor3D::lines().emplace_back(LineType {
        .p_w_i = p1_w,
        .p_w_j = p2_w,
        .color = RgbColor::kRed,
    });
    Visualizor3D::lines().emplace_back(LineType {
        .p_w_i = truth_line_w.GetPointOnLine(-1),
        .p_w_j = truth_line_w.GetPointOnLine(1),
        .color = RgbColor::kRed,
    });
    Visualizor3D::points().emplace_back(PointType {
        .p_w = truth_line_w.GetPointOnLine(0),
        .color = RgbColor::kGreen,
        .radius = 3,
    });
    // Draw result of triangulation.
    Visualizor3D::lines().emplace_back(LineType {
        .p_w_i = noised_line_w.ProjectPointOnLine(p1_w),
        .p_w_j = noised_line_w.ProjectPointOnLine(p2_w),
        .color = RgbColor::kPink,
    });
    Visualizor3D::lines().emplace_back(LineType {
        .p_w_i = estimated_line_w.ProjectPointOnLine(p1_w),
        .p_w_j = estimated_line_w.ProjectPointOnLine(p2_w),
        .color = RgbColor::kCyan,
    });
    while (!Visualizor3D::ShouldQuit()) {
        Visualizor3D::Refresh("Visualizor 3D", 20);
    }

    return 0;
}
