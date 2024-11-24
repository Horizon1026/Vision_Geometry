#include "line_triangulator.h"
#include "slam_basic_math.h"
#include "slam_operations.h"

namespace VISION_GEOMETRY {

bool LineTriangulator::Triangulate(const std::vector<Quat> &q_wc,
                                   const std::vector<Vec3> &p_wc,
                                   const std::vector<LineSegment2D> &line_in_norm_plane,
                                   LinePlucker3D &line_in_w) {
    RETURN_FALSE_IF(q_wc.size() < 2 || p_wc.size() < 2 || line_in_norm_plane.size() < 2);

    switch (options_.kMethod) {
        default:
        case TriangulationMethod::kAnalytic: {
            return TriangulateAnalytic(q_wc, p_wc, line_in_norm_plane, line_in_w);
        }
    }
    return false;
}

bool LineTriangulator::TriangulateAnalytic(const std::vector<Quat> &q_wc,
                                           const std::vector<Vec3> &p_wc,
                                           const std::vector<LineSegment2D> &line_in_norm_plane,
                                           LinePlucker3D &line_in_w) {

    return true;
}

}
