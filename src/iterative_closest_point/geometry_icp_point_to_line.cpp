#include "geometry_icp.h"
#include "kd_tree.h"

#include "math_kinematics.h"
#include "slam_operations.h"
#include "memory"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePoseByMethodPointToLine(const std::vector<Vec3> &all_ref_p_w,
                                                const std::vector<Vec3> &all_cur_p_w,
                                                Quat &q_cr,
                                                Vec3 &p_cr) {
    // Convert all reference points into kd-tree.
    std::vector<int32_t> sorted_point_indices(all_ref_p_w.size(), 0);
    for (uint32_t i = 0; i < sorted_point_indices.size(); ++i) {
        sorted_point_indices[i] = i;
    }
    std::unique_ptr<KdTreeNode<float, 3>> ref_kd_tree_ptr = std::make_unique<KdTreeNode<float, 3>>();
    ref_kd_tree_ptr->Construct(all_ref_p_w, sorted_point_indices, ref_kd_tree_ptr);

    return true;
}

}
