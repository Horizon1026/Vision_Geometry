#include "geometry_icp.h"
#include "kd_tree.h"
#include "nanoflann.h"
#include "slam_basic_math.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"
#include "memory"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePoseByMethodPointToLineWithNanoFlann(const std::vector<Vec3> &all_ref_p_w,
                                                             const std::vector<Vec3> &all_cur_p_w,
                                                             Quat &q_rc,
                                                             Vec3 &p_rc) {
    // Convert all reference points into kd-tree.
    NanoFlannKdTree ref_kd_tree(3, all_ref_p_w, 1);
    // Prepare something for knn search.
    const int32_t num_of_points_to_search = 2;
    std::vector<uint64_t> ret_indexes(num_of_points_to_search);
    std::vector<float> out_dists_sqr(num_of_points_to_search);
    nanoflann::KNNResultSet<float> search_result(num_of_points_to_search);
    const uint32_t index_step = GetIndexStep(all_cur_p_w.size());

    // Iterate to estimate relative pose between two point clouds.
    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        hessian.setZero();
        bias.setZero();

        // Iterate each current point to construct incremental function.
        for (uint32_t i = 0; i < all_cur_p_w.size(); i += index_step) {
            const Vec3 &cur_p_w = all_cur_p_w[i];
            const Vec3 transformed_cur_p_w = q_rc * cur_p_w + p_rc;
            search_result.init(&ret_indexes[0], &out_dists_sqr[0]);
            CONTINUE_IF(!ref_kd_tree.index->findNeighbors(search_result, &transformed_cur_p_w[0]));
            const Vec3 &ref_p_w_0 = all_ref_p_w[ret_indexes[0]];
            const Vec3 &ref_p_w_1 = all_ref_p_w[ret_indexes[1]];

            // Compute residual.
            const Vec3 diff_ref_p_w = ref_p_w_0 - ref_p_w_1;
            CONTINUE_IF(diff_ref_p_w.norm() < kZerofloat);
            const Vec3 residual = (transformed_cur_p_w - ref_p_w_0).cross(transformed_cur_p_w - ref_p_w_1) / diff_ref_p_w.norm();
            CONTINUE_IF(residual.norm() > options_.kMaxValidRelativePointDistance);

            // Compute jacobian.
            Mat3x6 jacobian = Mat3x6::Zero();
            jacobian.block<3, 3>(0, 0) = - Utility::SkewSymmetricMatrix(diff_ref_p_w) / diff_ref_p_w.norm();
            jacobian.block<3, 3>(0, 3) = - jacobian.block<3, 3>(0, 0) * Utility::SkewSymmetricMatrix(transformed_cur_p_w);

            // Construct hessian and bias.
            hessian += jacobian.transpose() * jacobian;
            bias -= jacobian.transpose() * residual;
        }

        // Solve incremental function.
        const Vec6 dx = hessian.ldlt().solve(bias);
        const Vec3 dp_rc = dx.head<3>();
        const Quat dq_rc = Quat(1, 0.5f * dx(3), 0.5f * dx(4), 0.5f * dx(5)).normalized();
        q_rc = q_rc * dq_rc;
        q_rc.normalize();
        p_rc += dp_rc;

        // Check if converged.
        BREAK_IF(dx.norm() < options_.kMaxConvergedStepLength);
    }

    return true;
}

bool IcpSolver::EstimatePoseByMethodPointToLineWithKdtree(const std::vector<Vec3> &all_ref_p_w,
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
    const int32_t num_of_points_to_search = 2;
    const uint32_t index_step = GetIndexStep(all_cur_p_w.size());

    // Iterate to estimate relative pose between two point clouds.
    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        hessian.setZero();
        bias.setZero();

        // Iterate each current point to construct incremental function.
        for (uint32_t i = 0; i < all_cur_p_w.size(); i += index_step) {
            const Vec3 &cur_p_w = all_cur_p_w[i];
            const Vec3 transformed_cur_p_w = q_rc * cur_p_w + p_rc;

            // Extract two points closest to target point.
            std::multimap<float, int32_t> result_of_nn_search;
            ref_kd_tree_ptr->SearchKnn(ref_kd_tree_ptr, all_ref_p_w, transformed_cur_p_w, num_of_points_to_search, result_of_nn_search);
            CONTINUE_IF(result_of_nn_search.size() != num_of_points_to_search);
            auto it = result_of_nn_search.begin();
            const Vec3 &ref_p_w_0 = all_ref_p_w[it->second];
            const Vec3 &ref_p_w_1 = all_ref_p_w[std::next(it)->second];
            CONTINUE_IF((ref_p_w_0 - transformed_cur_p_w).norm() > options_.kMaxValidRelativePointDistance);
            CONTINUE_IF((ref_p_w_1 - transformed_cur_p_w).norm() > options_.kMaxValidRelativePointDistance);

            // Compute residual.
            const Vec3 diff_ref_p_w = ref_p_w_0 - ref_p_w_1;
            CONTINUE_IF(diff_ref_p_w.norm() < kZerofloat);
            const Vec3 residual = (transformed_cur_p_w - ref_p_w_0).cross(transformed_cur_p_w - ref_p_w_1) / diff_ref_p_w.norm();

            // Compute jacobian.
            Mat3x6 jacobian = Mat3x6::Zero();
            jacobian.block<3, 3>(0, 0) = - Utility::SkewSymmetricMatrix(diff_ref_p_w) / diff_ref_p_w.norm();
            jacobian.block<3, 3>(0, 3) = - jacobian.block<3, 3>(0, 0) * Utility::SkewSymmetricMatrix(transformed_cur_p_w);

            // Construct hessian and bias.
            hessian += jacobian.transpose() * jacobian;
            bias -= jacobian.transpose() * residual;
        }

        // Solve incremental function.
        const Vec6 dx = hessian.ldlt().solve(bias);
        const Vec3 dp_rc = dx.head<3>();
        const Quat dq_rc = Quat(1, 0.5f * dx(3), 0.5f * dx(4), 0.5f * dx(5)).normalized();
        q_rc = q_rc * dq_rc;
        q_rc.normalize();
        p_rc += dp_rc;

        // Check if converged.
        BREAK_IF(dx.norm() < options_.kMaxConvergedStepLength);
    }

    return true;
}

}
