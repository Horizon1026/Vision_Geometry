#include "geometry_icp.h"
#include "slam_operations.h"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePose(const std::vector<Vec3> &all_ref_p_w,
                             const std::vector<Vec3> &all_cur_p_w,
                             Quat &q_rc,
                             Vec3 &p_rc) {
    RETURN_FALSE_IF(all_ref_p_w.empty() || all_cur_p_w.empty());

    switch (options_.kMethod) {
        default:
        case IcpMethod::kPointToPoint:
            return EstimatePoseByMethodPointToPoint(all_ref_p_w, all_cur_p_w, q_rc, p_rc);

        case IcpMethod::kPointToLine:
            return EstimatePoseByMethodPointToLine(all_ref_p_w, all_cur_p_w, q_rc, p_rc);

        case IcpMethod::kPointToPlane:
            // TODO:
            break;
    }

    return true;
}

}
