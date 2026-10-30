#ifndef _GEOMETRY_HOMOGRAPHY_PNP_H_
#define _GEOMETRY_HOMOGRAPHY_PNP_H_

#include "basic_type.h"

namespace vision_geometry {

/* Class HomographyPnpSolver Declaration. */
class HomographyPnpSolver {

public:
    enum class Method : uint8_t {
        kOptimize = 0,
        kOptimizeRansac = 1,
        kOptimizeHuber = 2,
        kOptimizeCauchy = 3,
    };

    enum class Result : uint8_t {
        kUnsolved = 0,
        kSolved = 1,
        kLargeResidual = 2,
    };

    struct Options {
        Method kMethod = Method::kOptimize;
        int32_t kMaxIteration = 100;
        float kMaxConvergeStep = 1e-6f;
        float kDefaultHuberKernelParameter = 0.01f;
        float kDefaultCauchyKernelParameter = 0.01f;
    };

public:
    HomographyPnpSolver() = default;
    virtual ~HomographyPnpSolver() = default;

    bool EstimatePose(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz, Vec3 &p_cr,
                      Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool Project(const Vec3 &ref_bearing_xyz, const float ref_depth_m, const Vec2 &homo_plane_slope_xy, const Vec3 &p_cr, const Quat &q_cr,
                 Vec3 &cur_bearing_xyz);
    bool EstimatePoseUseAll(const std::vector<Vec3> all_ref_bearing_xyz, const float ref_depth_m, const std::vector<Vec3> &all_cur_bearing_xyz, Vec3 &p_cr,
                            Quat &q_cr, Vec2 &homo_plane_slope_xy, std::vector<uint8_t> &status);

    // Huber weight function: w = 1 if |r| <= delta, w = delta/|r| if |r| > delta.
    float Huber(float param, float x);
    float Cauchy(float param, float x);

private:
    Options options_;
};

}  // namespace vision_geometry

#endif  // end of _GEOMETRY_HOMOGRAPHY_PNP_H_
