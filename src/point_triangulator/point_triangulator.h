#ifndef _VISION_GEOMETRY_POINT_TRIANGULATOR_H_
#define _VISION_GEOMETRY_POINT_TRIANGULATOR_H_

#include "basic_type.h"

namespace VISION_GEOMETRY {

/* Class PointTriangulator Declaration. */
class PointTriangulator {

public:
    enum class TriangulationMethod: uint8_t {
        kAnalytic = 0,
        kIterative = 1,
    };

    struct TriangulationOptions {
        uint32_t kMaxIteration = 10;
        uint32_t kMaxUsedCameraView = 10;
        float kMinValidDepth = 1e-3f;
        float kMaxConvergeStep = 1e-6f;
        TriangulationMethod kMethod = TriangulationMethod::kAnalytic;
    };

public:
    PointTriangulator() = default;
    virtual ~PointTriangulator() = default;

    bool Triangulate(const std::vector<Quat> &q_wc,
                     const std::vector<Vec3> &p_wc,
                     const std::vector<Vec2> &norm_uv,
                     Vec3 &p_w);

    static float GetParallexAngle(const Quat &q_wci, const Vec3 &p_wci,
                                  const Quat &q_wcj, const Vec3 &p_wcj,
                                  const Vec2 &norm_xy_i, const Vec2 &norm_xy_j);

    // Reference for member variables.
    TriangulationOptions &options() { return options_; }
    // Const reference for member variables.
    const TriangulationOptions &options() const { return options_; }

private:
    bool TriangulateAnalytic(const std::vector<Quat> &q_wc,
                             const std::vector<Vec3> &p_wc,
                             const std::vector<Vec2> &norm_uv,
                             Vec3 &p_w);
    bool TriangulateIterative(const std::vector<Quat> &q_wc,
                              const std::vector<Vec3> &p_wc,
                              const std::vector<Vec2> &norm_uv,
                              Vec3 &p_w);
    bool CheckDepthInMultiView(const std::vector<Quat> &q_wc,
                               const std::vector<Vec3> &p_wc,
                               const Vec3 &p_w);

private:
    TriangulationOptions options_;
};

}

#endif // end of _VISION_GEOMETRY_POINT_TRIANGULATOR_H_
