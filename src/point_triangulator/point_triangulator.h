#ifndef _VISION_GEOMETRY_POINT_TRIANGULATOR_H_
#define _VISION_GEOMETRY_POINT_TRIANGULATOR_H_

#include "basic_type.h"

namespace vision_geometry {

/* Class PointTriangulator Declaration. */
class PointTriangulator {

public:
    enum class Method : uint8_t {
        kAnalytic = 0,
        kOptimize = 1,
        kOptimizeHuber = 2,
        kOptimizeCauchy = 3,
    };

    struct Options {
        uint32_t kMaxIteration = 10;
        uint32_t kMaxUsedCameraView = 10;
        float kMinValidDepth = 1e-3f;
        float kMaxConvergeStep = 1e-6f;
        float kMaxToleranceReprojectionError = 0.1f;
        float kDefaultHuberKernelParameter = 1.0f;
        float kDefaultCauchyKernelParameter = 1.0f;
        Method kMethod = Method::kAnalytic;
    };

public:
    PointTriangulator() = default;
    virtual ~PointTriangulator() = default;

    bool Triangulate(const std::vector<Quat> &q_wc, const std::vector<Vec3> &p_wc, const std::vector<Vec2> &norm_xy, Vec3 &p_w);

    static float GetSineOfParallexAngle(const Quat &q_wci, const Vec3 &p_wci, const Quat &q_wcj, const Vec3 &p_wcj, const Vec2 &norm_xy_i,
                                        const Vec2 &norm_xy_j);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool TriangulateAnalytic(const std::vector<Quat> &q_wc, const std::vector<Vec3> &p_wc, const std::vector<Vec2> &norm_xy, Vec3 &p_w);
    bool TriangulateIterative(const std::vector<Quat> &q_wc, const std::vector<Vec3> &p_wc, const std::vector<Vec2> &norm_xy, Vec3 &p_w);
    bool CheckResultInMultiView(const std::vector<Quat> &q_wc, const std::vector<Vec3> &p_wc, const std::vector<Vec2> &norm_xy, const Vec3 &p_w);
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

}  // namespace vision_geometry

#endif  // end of _VISION_GEOMETRY_POINT_TRIANGULATOR_H_
