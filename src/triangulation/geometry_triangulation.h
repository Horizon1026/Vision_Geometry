#ifndef _GEOMETRY_TRIANGULATION_H_
#define _GEOMETRY_TRIANGULATION_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

/* Class Triangulator Declaration. */
class Triangulator {

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
    explicit Triangulator() = default;
    virtual ~Triangulator() = default;

    bool Triangulate(const std::vector<Quat> &q_wc,
                     const std::vector<Vec3> &p_wc,
                     const std::vector<Vec2> &norm_uv,
                     Vec3 &p_w);

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

#endif
