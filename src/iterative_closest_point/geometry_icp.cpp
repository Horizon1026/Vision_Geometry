#include "geometry_icp.h"
#include "slam_operations.h"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePose(const std::vector<Vec3> &all_ref_p_w,
                             const std::vector<Vec3> &all_cur_p_w,
                             Quat &q_cr,
                             Vec3 &p_cr) {
    RETURN_FALSE_IF(all_ref_p_w.empty() || all_cur_p_w.empty());

    switch (options_.kMethod) {
        default:
        case IcpMethod::kPointToPoint:
            return EstimatePoseByMethodPointToPoint(all_ref_p_w, all_cur_p_w, q_cr, p_cr);

        case IcpMethod::kPointToLine:
            // TODO:
            break;

        case IcpMethod::kPointToPlane:
            // TODO:
            break;
    }

    return true;
}

}
