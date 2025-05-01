#ifndef _VISION_GEOMETRY_POINT_TRIANGULATOR_H_
#define _VISION_GEOMETRY_POINT_TRIANGULATOR_H_

#include "basic_type.h"

namespace VISION_GEOMETRY {

/* Class PointTriangulator Declaration. */
class PointTriangulator {

public:
    enum class Method: uint8_t {
        kAnalytic = 0,
        kIterative = 1,
    };

    struct Options {
        uint32_t kMaxIteration = 10;
        uint32_t kMaxUsedCameraView = 10;
        float kMinValidDepth = 1e-3f;
        float kMaxConvergeStep = 1e-6f;
        float kMaxToleranceReprojectionError = 0.1f;
        Method kMethod = Method::kAnalytic;
    };

public:
    PointTriangulator() = default;
    virtual ~PointTriangulator() = default;

    bool Triangulate(const std::vector<Quat> &q_wc,
                     const std::vector<Vec3> &p_wc,
                     const std::vector<Vec2> &norm_xy,
                     Vec3 &p_w);

    static float GetSineOfParallexAngle(const Quat &q_wci, const Vec3 &p_wci,
                                        const Quat &q_wcj, const Vec3 &p_wcj,
                                        const Vec2 &norm_xy_i, const Vec2 &norm_xy_j);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool TriangulateAnalytic(const std::vector<Quat> &q_wc,
                             const std::vector<Vec3> &p_wc,
                             const std::vector<Vec2> &norm_xy,
                             Vec3 &p_w);
    bool TriangulateIterative(const std::vector<Quat> &q_wc,
                              const std::vector<Vec3> &p_wc,
                              const std::vector<Vec2> &norm_xy,
                              Vec3 &p_w);
    bool CheckResultInMultiView(const std::vector<Quat> &q_wc,
                                const std::vector<Vec3> &p_wc,
                                const std::vector<Vec2> &norm_xy,
                                const Vec3 &p_w);

private:
    Options options_;
};

}

#endif // end of _VISION_GEOMETRY_POINT_TRIANGULATOR_H_
