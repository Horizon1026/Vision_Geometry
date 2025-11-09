#ifndef _VISION_GEOMETRY_LINE_TRIANGULATOR_H_
#define _VISION_GEOMETRY_LINE_TRIANGULATOR_H_

#include "basic_type.h"
#include "line_segment.h"

namespace VISION_GEOMETRY {

/* Class LineTriangulator Declaration. */
class LineTriangulator {

public:
    enum class Method : uint8_t {
        kAnalytic = 0,
        kOrdinaryLeastSquare = 1,
        kOptimize = 2,
    };

    struct Options {
        uint32_t kMaxIteration = 10;
        uint32_t kMaxUsedCameraView = 10;
        float kMaxConvergeStep = 1e-6f;
        Method kMethod = Method::kAnalytic;
    };

public:
    LineTriangulator() = default;
    virtual ~LineTriangulator() = default;

    bool Triangulate(const std::vector<Quat> &all_q_wc, const std::vector<Vec3> &all_p_wc, const std::vector<LineSegment2D> &lines_in_norm_plane,
                     LinePlucker3D &plucker_in_w);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool TriangulateAnalytic(const std::vector<Quat> &all_q_wc, const std::vector<Vec3> &all_p_wc, const std::vector<LineSegment2D> &lines_in_norm_plane,
                             LinePlucker3D &plucker_in_w);
    bool TriangulateIterative(const std::vector<Quat> &all_q_wc, const std::vector<Vec3> &all_p_wc, const std::vector<LineSegment2D> &lines_in_norm_plane,
                              LinePlucker3D &plucker_in_w);


private:
    Options options_;
};

}  // namespace VISION_GEOMETRY

#endif  // end of _VISION_GEOMETRY_LINE_TRIANGULATOR_H_
