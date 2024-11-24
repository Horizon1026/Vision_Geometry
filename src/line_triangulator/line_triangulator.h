#ifndef _VISION_GEOMETRY_LINE_TRIANGULATOR_H_
#define _VISION_GEOMETRY_LINE_TRIANGULATOR_H_

#include "basic_type.h"
#include "line_segment.h"

namespace VISION_GEOMETRY {

/* Class LineTriangulator Declaration. */
class LineTriangulator {

public:
    enum class TriangulationMethod: uint8_t {
        kAnalytic = 0,
        kOrdinaryLeastSquare = 1,
        kIterative = 2,
    };

    struct TriangulationOptions {
        TriangulationMethod kMethod = TriangulationMethod::kAnalytic;
    };

public:
    LineTriangulator() = default;
    virtual ~LineTriangulator() = default;

    bool Triangulate(const std::vector<Quat> &q_wc,
                     const std::vector<Vec3> &p_wc,
                     const std::vector<LineSegment2D> &line_in_norm_plane,
                     LinePlucker3D &line_in_w);

    // Reference for member variables.
    TriangulationOptions &options() { return options_; }
    // Const reference for member variables.
    const TriangulationOptions &options() const { return options_; }

private:
    bool TriangulateAnalytic(const std::vector<Quat> &q_wc,
                             const std::vector<Vec3> &p_wc,
                             const std::vector<LineSegment2D> &line_in_norm_plane,
                             LinePlucker3D &line_in_w);

private:
    TriangulationOptions options_;
};

}

#endif // end of _VISION_GEOMETRY_LINE_TRIANGULATOR_H_
