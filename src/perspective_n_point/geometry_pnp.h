#ifndef _GEOMETRY_PNP_H_
#define _GEOMETRY_PNP_H_

#include "basic_type.h"

namespace VISION_GEOMETRY {

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
        uint32_t kMaxIteration = 10;
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

    bool EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Quat &q_wc, Vec3 &p_wc, std::vector<uint8_t> &status);
    void CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Quat &q_wc, Vec3 &p_wc, std::vector<uint8_t> &status);
    bool EstimatePose(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy, Quat &q_wi,
                      Vec3 &p_wi, std::vector<uint8_t> &status);
    void CheckPnpStatus(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                        Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status);

    // Reference for member variables.
    Options &options() { return options_; }

    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Quat &q_wc, Vec3 &p_wc, std::vector<uint8_t> &status);
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Quat &q_wc, Vec3 &p_wc);
    bool EstimatePoseRansac(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Quat &q_wc, Vec3 &p_wc, std::vector<uint8_t> &status);
    bool EstimatePoseDlt(const std::vector<Vec3> &p_w, const std::vector<Vec2> &norm_xy, Quat &q_wc, Vec3 &p_wc, std::vector<uint8_t> &status);

    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                            Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status);
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                            Quat &q_wi, Vec3 &p_wi);
    bool EstimatePoseRansac(const std::vector<Vec3> &p_w, const std::vector<Quat> &q_ic, const std::vector<Vec3> &p_ic, const std::vector<Vec2> &norm_xy,
                            Quat &q_wi, Vec3 &p_wi, std::vector<uint8_t> &status);

    inline float Huber(float param, float x) {
        float huber = 1.0f;
        if (x > param) {
            huber = 2.0f * std::sqrt(x) * param - param * param;
            huber /= x;
        }
        return huber;
    }

    inline float Cauchy(float param, float x) {
        float cauchy = 1.0f;
        float param2 = param * param;
        if (x > param) {
            cauchy = param2 * std::log(1.0f / param2 + 1.0f);
            cauchy /= x;
        }
        return cauchy;
    }

private:
    Options options_;
};
}  // namespace VISION_GEOMETRY

#endif  // end of _GEOMETRY_PNP_H_
