#include "geometry_icp.h"
#include "kd_tree.h"
#include "nanoflann.h"
#include "plane.h"
#include "slam_basic_math.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"
#include "memory"

namespace VISION_GEOMETRY {

bool IcpSolver::EstimatePoseByMethodPointToPlaneWithNanoFlann(const std::vector<Vec3> &all_ref_p_w,
                                                              const std::vector<Vec3> &all_cur_p_w,
                                                              Quat &q_rc,
                                                              Vec3 &p_rc) {
    const int32_t num_of_points_to_search = 5;

    // Convert all reference points into kd-tree.
    NanoFlannKdTree ref_kd_tree(3, all_ref_p_w, 1);
    // Prepare something for knn search.
    std::vector<size_t> ret_indexes(num_of_points_to_search);
    std::vector<float> out_dists_sqr(num_of_points_to_search);
    nanoflann::KNNResultSet<float> search_result(num_of_points_to_search);
    std::vector<Vec3> searched_points;
    searched_points.reserve(num_of_points_to_search);

    // Iterate to estimate relative pose between two point clouds.
    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        hessian.setZero();
        bias.setZero();

        // Iterate each current point to construct incremental function.
        for (const auto &cur_p_w : all_cur_p_w) {
            const Vec3 transformed_cur_p_w = q_rc * cur_p_w + p_rc;

            // Extract three points closest to target point.
            search_result.init(&ret_indexes[0], &out_dists_sqr[0]);
            CONTINUE_IF(!ref_kd_tree.index->findNeighbors(search_result, &transformed_cur_p_w[0]));
            searched_points.clear();
            for (uint32_t i = 0; i < ret_indexes.size(); ++i) {
                CONTINUE_IF(out_dists_sqr[i] > options_.kMaxValidRelativePointDistance);
                searched_points.emplace_back(all_ref_p_w[ret_indexes[i]]);
            }

            // Fit plane model.
            Plane3D plane;
            CONTINUE_IF(!plane.FitPlaneModel(searched_points));

            // Compute residual.
            const float residual = plane.GetDistanceToPlane(transformed_cur_p_w);

            // Compute jacobian.
            Mat1x6 jacobian = Mat1x6::Zero();
            jacobian.block<1, 3>(0, 0) = plane.normal_vector().transpose();
            jacobian.block<1, 3>(0, 3) = - plane.normal_vector().transpose() * Utility::SkewSymmetricMatrix(transformed_cur_p_w);

            // Construct hessian and bias.
            hessian += jacobian.transpose() * jacobian;
            bias -= jacobian.transpose() * residual;
        }

        // Solve incremental function.
        const Vec6 dx = hessian.ldlt().solve(bias);
        const Vec3 dp_rc = dx.head<3>();
        p_rc += dp_rc;
        const Quat dq_rc = Utility::DeltaQ(dx.tail<3>());
        q_rc = q_rc * dq_rc;
        q_rc.normalize();

        // Check if converged.
        BREAK_IF(dx.norm() < options_.kMaxConvergedStepLength);
    }

    return true;
}

bool IcpSolver::EstimatePoseByMethodPointToPlaneWithKdtree(const std::vector<Vec3> &all_ref_p_w,
                                                           const std::vector<Vec3> &all_cur_p_w,
                                                           Quat &q_rc,
                                                           Vec3 &p_rc) {
    const int32_t num_of_points_to_search = 5;
    // Convert all reference points into kd-tree.
    std::vector<int32_t> sorted_point_indices(all_ref_p_w.size(), 0);
    for (uint32_t i = 0; i < sorted_point_indices.size(); ++i) {
        sorted_point_indices[i] = i;
    }
    std::unique_ptr<KdTreeNode<float, 3>> ref_kd_tree_ptr = std::make_unique<KdTreeNode<float, 3>>();
    ref_kd_tree_ptr->Construct(all_ref_p_w, sorted_point_indices, ref_kd_tree_ptr);
    std::vector<Vec3> searched_points;
    searched_points.reserve(num_of_points_to_search);

    // Iterate to estimate relative pose between two point clouds.
    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        hessian.setZero();
        bias.setZero();

        // Iterate each current point to construct incremental function.
        for (const auto &cur_p_w : all_cur_p_w) {
            const Vec3 transformed_cur_p_w = q_rc * cur_p_w + p_rc;

            // Extract three points closest to target point.
            std::multimap<float, int32_t> result_of_nn_search;
            ref_kd_tree_ptr->SearchKnn(ref_kd_tree_ptr, all_ref_p_w, transformed_cur_p_w, num_of_points_to_search, result_of_nn_search);
            CONTINUE_IF(result_of_nn_search.size() != num_of_points_to_search);

            searched_points.clear();
            for (const auto &[distance, index]: result_of_nn_search) {
                CONTINUE_IF(distance > options_.kMaxValidRelativePointDistance);
                searched_points.emplace_back(all_ref_p_w[index]);
            }

            // Fit plane model.
            Plane3D plane;
            CONTINUE_IF(!plane.FitPlaneModel(searched_points));

            // Compute residual.
            const float residual = plane.GetDistanceToPlane(transformed_cur_p_w);

            // Compute jacobian.
            Mat1x6 jacobian = Mat1x6::Zero();
            jacobian.block<1, 3>(0, 0) = plane.normal_vector().transpose();
            jacobian.block<1, 3>(0, 3) = - plane.normal_vector().transpose() * Utility::SkewSymmetricMatrix(transformed_cur_p_w);

            // Construct hessian and bias.
            hessian += jacobian.transpose() * jacobian;
            bias -= jacobian.transpose() * residual;
        }

        // Solve incremental function.
        const Vec6 dx = hessian.ldlt().solve(bias);
        const Vec3 dp_rc = dx.head<3>();
        p_rc += dp_rc;
        const Quat dq_rc = Utility::DeltaQ(dx.tail<3>());
        q_rc = q_rc * dq_rc;
        q_rc.normalize();

        // Check if converged.
        BREAK_IF(dx.norm() < options_.kMaxConvergedStepLength);
    }

    return true;
}

}
