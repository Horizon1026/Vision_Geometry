#include "geometry_triangulation.h"
#include "math_kinematics.h"

namespace VISION_GEOMETRY {

bool Triangulator::Triangulate(const std::vector<Quat> &q_wc,
                               const std::vector<Vec3> &p_wc,
                               const std::vector<Vec2> &norm_uv,
                               Vec3 &p_w) {
    switch (options_.kMethod) {
        default:
        case TriangulationMethod::ANALYTIC: {
            return TriangulateAnalytic(q_wc, p_wc, norm_uv, p_w);
        }
        case TriangulationMethod::ITERATIVE: {
            return TriangulateIterative(q_wc, p_wc, norm_uv, p_w);
        }
    }
    return false;
}

bool Triangulator::TriangulateAnalytic(const std::vector<Quat> &q_wc,
                                       const std::vector<Vec3> &p_wc,
                                       const std::vector<Vec2> &norm_uv,
                                       Vec3 &p_w) {
    if (q_wc.size() < 2) {
        return false;
    }

    uint32_t used_camera_num = options_.kMaxUsedCameraView < q_wc.size() ? options_.kMaxUsedCameraView : q_wc.size();

    Eigen::Matrix<float, Eigen::Dynamic, 4> A;
    A.resize(used_camera_num * 2, 4);

    for (uint32_t i = 0; i < used_camera_num; ++i) {
        auto pose = Utility::TransformMatrix<float>(q_wc[i].inverse(), - (q_wc[i].inverse() * p_wc[i]));
        A.row(2 * i) = norm_uv[i][0] * pose.row(2) - pose.row(0);
        A.row(2 * i + 1) = norm_uv[i][1] * pose.row(2) - pose.row(1);
    }
    Vec4 x = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    if (std::fabs(x(3)) < kZero) {
        return false;
    }
    p_w = x.head<3>() / x(3);

    return CheckDepthInMultiView(q_wc, p_wc, p_w);
}

bool Triangulator::TriangulateIterative(const std::vector<Quat> &q_wc,
                                        const std::vector<Vec3> &p_wc,
                                        const std::vector<Vec2> &norm_uv,
                                        Vec3 &p_w) {
    if (q_wc.size() != p_wc.size() || q_wc.size() != norm_uv.size()) {
        return false;
    }

    uint32_t used_camera_num = options_.kMaxUsedCameraView < q_wc.size() ? options_.kMaxUsedCameraView : q_wc.size();

    Mat3 H;
    Vec3 b;
    Mat2x3 jacobian;

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        H.setZero();
        b.setZero();

        for (uint32_t i = 0; i < used_camera_num; ++i) {
            Vec3 p_c = q_wc[i].inverse() * (p_w - p_wc[i]);
            if (p_c.z() < options_.kMinValidDepth || std::isnan(p_c.z())) {
                continue;
            }

            const float inv_depth = 1.0f / p_c.z();
            const float inv_depth2 = inv_depth * inv_depth;

            Vec2 residual = Vec2(p_c(0) / p_c(2), p_c(1) / p_c(2)) - norm_uv[i];

            Mat2x3 jacobian_2d_3d;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth2,
                              0, inv_depth, - p_c(1) * inv_depth2;

            jacobian = jacobian_2d_3d * q_wc[i].inverse().matrix();

            H += jacobian.transpose() * jacobian;
            b -= jacobian.transpose() * residual;
        }

        Vec3 dx = H.ldlt().solve(b);

        float norm_dx = dx.squaredNorm();
        if (std::isnan(norm_dx) == true) {
            return false;
        }
        norm_dx = std::sqrt(norm_dx);

        p_w += dx;

        if (norm_dx < options_.kMaxConvergeStep) {
            break;
        }
    }

    return CheckDepthInMultiView(q_wc, p_wc, p_w);
}

bool Triangulator::CheckDepthInMultiView(const std::vector<Quat> &q_wc,
                                         const std::vector<Vec3> &p_wc,
                                         const Vec3 &p_w) {
    for (uint32_t i = 0; i < q_wc.size(); ++i) {
        Vec3 p_c = q_wc[i].inverse() * (p_w - p_wc[i]);
        if (p_c.z() < 0) {
            return false;
        }
    }

    return true;
}

}
