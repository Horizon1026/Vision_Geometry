#include "geometry_icp.h"
#include "kd_tree.h"

#include "math_kinematics.h"
#include "slam_operations.h"
#include "memory"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePoseByMethodPointToPoint(const std::vector<Vec3> &all_ref_p_w,
                                                 const std::vector<Vec3> &all_cur_p_w,
                                                 Quat &q_rc,
                                                 Vec3 &p_rc) {
    // Convert all reference points into kd-tree.
    std::vector<int32_t> sorted_point_indices(all_ref_p_w.size(), 0);
    for (uint32_t i = 0; i < sorted_point_indices.size(); ++i) {
        sorted_point_indices[i] = i;
    }
    std::unique_ptr<KdTreeNode<float, 3>> ref_kd_tree_ptr = std::make_unique<KdTreeNode<float, 3>>();
    ref_kd_tree_ptr->Construct(all_ref_p_w, sorted_point_indices, ref_kd_tree_ptr);

    // Prepare for iteration.
    q_rc.normalize();
    std::vector<Vec3> sub_ref_p_w = all_ref_p_w;
    std::vector<Vec3> sub_cur_p_w = all_cur_p_w;

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Transform current points by current relative pose.
        sub_ref_p_w.clear();
        sub_cur_p_w.clear();

        // Extract relative point pairs by kd-tree.
        for (const auto &cur_p_w : all_cur_p_w) {
            const Vec3 transformed_cur_p_w = q_rc * cur_p_w + p_rc;
            std::multimap<float, int32_t> result_of_nn_search;
            ref_kd_tree_ptr->SearchKnn(ref_kd_tree_ptr, all_ref_p_w, transformed_cur_p_w, 1, result_of_nn_search);
            const auto &pair = *result_of_nn_search.begin();
            CONTINUE_IF(pair.first > options_.kMaxValidRelativePointDistance);

            const Vec3 &nearest_ref_p_w = all_ref_p_w[pair.second];
            sub_ref_p_w.emplace_back(nearest_ref_p_w);
            sub_cur_p_w.emplace_back(cur_p_w);
        }

        if (sub_ref_p_w.size() < 3) {
            ReportWarn("[ICP] Failed to estimate relative pose for lack of relative points.");
            return false;
        }

        Quat d_q_rc = q_rc;
        Vec3 d_p_rc = p_rc;
        RETURN_FALSE_IF(!EstimatePoseByPointPairs(sub_ref_p_w, sub_cur_p_w, q_rc, p_rc));

        d_q_rc = d_q_rc.inverse() * q_rc;
        d_p_rc = p_rc - d_p_rc;
        BREAK_IF(d_q_rc.vec().norm() + d_p_rc.norm() < 1e-6f);
    }

    return true;
}

bool IcpSolver::EstimatePoseByPointPairs(const std::vector<Vec3> &all_ref_p_w,
                                         const std::vector<Vec3> &all_cur_p_w,
                                         Quat &q_rc,
                                         Vec3 &p_rc) {
    RETURN_FALSE_IF(all_ref_p_w.empty());
    RETURN_FALSE_IF(all_ref_p_w.size() != all_cur_p_w.size());
    const int32_t size = static_cast<int32_t>(all_ref_p_w.size());

    // Compute centroids of two point clouds.
    Vec3 ref_mid = Vec3::Zero();
    Vec3 cur_mid = Vec3::Zero();
    for (const auto &p_w : all_ref_p_w) {
        ref_mid += p_w;
    }
    for (const auto &p_w : all_cur_p_w) {
        cur_mid += p_w;
    }
    ref_mid /= static_cast<float>(size);
    cur_mid /= static_cast<float>(size);

    // Compute covariance.
    Mat3 covariance = Mat3::Zero();
    for (int32_t i = 0; i < size; ++i) {
        const Vec3 ref_point_diff = all_ref_p_w[i] - ref_mid;
        const Vec3 cur_point_diff = all_cur_p_w[i] - cur_mid;
        covariance += cur_point_diff * ref_point_diff.transpose();
    }

    // Decompose covariance to sove relative pose.
    Eigen::JacobiSVD<Mat3> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Mat3 R_rc = svd.matrixV() * svd.matrixU().transpose();
    q_rc = Quat(R_rc).normalized();
    if (R_rc.determinant() < 0) {
        Mat3 matrix_v = svd.matrixV();
        matrix_v(2, 2) = - matrix_v(2, 2);
        R_rc = matrix_v * svd.matrixU().transpose();
        q_rc = Quat(R_rc).normalized();
    }
    p_rc = ref_mid - q_rc * cur_mid;

    return true;
}

}
