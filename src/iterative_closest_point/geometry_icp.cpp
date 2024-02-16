#include "geometry_icp.h"
#include "kd_tree.h"

#include "math_kinematics.h"
#include "slam_operations.h"
#include "memory"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePose(const std::vector<Vec3> &all_ref_p_w,
                             const std::vector<Vec3> &all_cur_p_w,
                             Quat &q_cr,
                             Vec3 &p_cr) {
    RETURN_FALSE_IF(all_ref_p_w.empty() || all_cur_p_w.empty());

    // Convert reference points into kd-tree.
    std::vector<int32_t> sorted_point_indices(all_cur_p_w.size(), 0);
    for (uint32_t i = 0; i < sorted_point_indices.size(); ++i) {
        sorted_point_indices[i] = i;
    }
    std::unique_ptr<KdTreeNode<float, 3>> cur_kd_tree_ptr = std::make_unique<KdTreeNode<float, 3>>();
    cur_kd_tree_ptr->Construct(all_cur_p_w, sorted_point_indices, cur_kd_tree_ptr);

    // Prepare for iteration.
    q_cr.normalize();
    std::vector<Vec3> sub_ref_p_w = all_ref_p_w;
    std::vector<Vec3> sub_cur_p_w = all_cur_p_w;

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Transform current points by current relative pose.
        sub_ref_p_w.clear();
        sub_cur_p_w.clear();

        // Extract relative point pairs by kd-tree.
        for (const auto &ref_p_w : all_ref_p_w) {
            const Vec3 transformed_ref_p_w = q_cr * ref_p_w + p_cr;
            std::multimap<float, int32_t> result_of_nn_search;
            cur_kd_tree_ptr->SearchKnn(cur_kd_tree_ptr, all_cur_p_w, transformed_ref_p_w, 1, result_of_nn_search);
            const auto &pair = *result_of_nn_search.begin();
            CONTINUE_IF(pair.first > options_.kMaxValidRelativePointDistance);

            const Vec3 &nearest_cur_p_w = all_cur_p_w[pair.second];
            sub_ref_p_w.emplace_back(ref_p_w);
            sub_cur_p_w.emplace_back(nearest_cur_p_w);
        }

        if (sub_ref_p_w.size() < 3) {
            ReportWarn("[ICP] Failed to estimate relative pose for lack of relative points.");
            return false;
        }

        Quat d_q_cr = q_cr;
        Vec3 d_p_cr = p_cr;
        RETURN_FALSE_IF(!EstimatePoseByPairs(sub_ref_p_w, sub_cur_p_w, q_cr, p_cr));

        d_q_cr = d_q_cr.inverse() * q_cr;
        d_p_cr = p_cr - d_p_cr;
        BREAK_IF(d_q_cr.vec().norm() + d_p_cr.norm() < 1e-6f);
    }

    return true;
}

bool IcpSolver::EstimatePoseByPairs(const std::vector<Vec3> &all_ref_p_w,
                                    const std::vector<Vec3> &all_cur_p_w,
                                    Quat &q_cr,
                                    Vec3 &p_cr) {
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
        covariance += ref_point_diff * cur_point_diff.transpose();
    }

    // Decompose covariance to sove relative pose.
    Eigen::JacobiSVD<Mat3> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Mat3 R_cr = svd.matrixV() * svd.matrixU().transpose();
    q_cr = Quat(R_cr).normalized();
    if (R_cr.determinant() < 0) {
        Mat3 matrix_v = svd.matrixV();
        matrix_v(2, 2) = - matrix_v(2, 2);
        R_cr = matrix_v * svd.matrixU().transpose();
        q_cr = Quat(R_cr).normalized();
    }
    p_cr = cur_mid - q_cr * ref_mid;

    return true;
}

}
