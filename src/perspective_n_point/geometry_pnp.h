#ifndef _GEOMETRY_PNP_H_
#define _GEOMETRY_PNP_H_

#include "basic_type.h"

namespace vision_geometry {

/* Class PnpSolver Declaration. */
class PnpSolver {

public:
    enum class Method : uint8_t {
        kOptimize = 0,
        kOptimizeRansac = 1,
        kOptimizeHuber = 2,
        kOptimizeCauchy = 3,
        kDirectLinearTransform = 4,
    };

    enum class Result : uint8_t {
        kUnsolved = 0,
        kSolved = 1,
        kLargeResidual = 2,
    };

    struct Options {
        uint32_t kMaxSolvePointsNumber = 200;
        uint32_t kMaxIteration = 50;
        float kMaxConvergeStep = 1e-6f;
        float kMaxConvergeResidual = 1e-3f;
        float kMinRansacInlierRatio = 0.9f;
        float kMaxPnpResidual = 1e-3f;
        float kMinValidDepth = 1e-3f;
        float kDefaultHuberKernelParameter = 1.0f;
        float kDefaultCauchyKernelParameter = 1.0f;
        Method kMethod = Method::kOptimizeRansac;
    };

public:
    explicit PnpSolver() = default;
    virtual ~PnpSolver() = default;

    // Bearing overloads normalize camera-frame rays without projecting onto the image plane.

    // For RANSAC, a nonempty status selects kSolved (tracked) pairs; other entries are preserved.
    bool EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);
    bool EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);
    void CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);
    void CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);
    float ComputeAverageReprojectionError(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, const Vec3 &p_wc, const Quat &q_wc) const;
    float ComputeAverageReprojectionError(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, const Vec3 &p_wc, const Quat &q_wc) const;
    bool EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec2> &norm_xy, Vec3 &p_wi,
                      Quat &q_wi, std::vector<uint8_t> &status);
    bool EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wi,
                      Quat &q_wi, std::vector<uint8_t> &status);
    void CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec2> &norm_xy,
                        Vec3 &p_wi, Quat &q_wi, std::vector<uint8_t> &status);
    void CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec3> &bearing_xyz,
                        Vec3 &p_wi, Quat &q_wi, std::vector<uint8_t> &status);

    // Reference for member variables.
    Options &options() { return options_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Vec3 &p_wc, Quat &q_wc);
    bool EstimatePoseRansac(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);
    bool EstimatePoseDlt(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Vec3 &p_wc, Quat &q_wc, std::vector<uint8_t> &status);

    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec2> &norm_xy,
                            Vec3 &p_wi, Quat &q_wi, std::vector<uint8_t> &status);
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec2> &norm_xy,
                            Vec3 &p_wi, Quat &q_wi);
    bool EstimatePoseRansac(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic, const std::vector<Vec2> &norm_xy,
                            Vec3 &p_wi, Quat &q_wi, std::vector<uint8_t> &status);

    bool OptimizeBearings(const std::vector<Vec3> &p_w, const std::vector<Vec3> &p_ic, const std::vector<Quat> &q_ic,
                          const std::vector<Vec3> &bearing_xyz, Vec3 &p_wi, Quat &q_wi);
    bool EstimatePoseDlt(const std::vector<Vec3> &p_w, const std::vector<Vec3> &bearing_xyz, Vec3 &p_wc, Quat &q_wc);

    bool SolveDltConstraint(const Mat &A, Mat3x4 &P, bool check_rank);

    // Huber weight function: w = 1 if |r| <= delta, w = delta/|r| if |r| > delta.
    float Huber(float param, float x);
    float Cauchy(float param, float x);

private:
    Options options_;
};
}  // namespace vision_geometry

#endif  // end of _GEOMETRY_PNP_H_
