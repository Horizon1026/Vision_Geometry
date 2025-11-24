#include "geometry_ndt.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace vision_geometry {

bool NdtSolver::Initialize() {
    ref_voxels_.options().kRadius = Vec3(options_.kMaxLidarScanRadius, options_.kMaxLidarScanRadius, options_.kMaxLidarScanRadius);
    ref_voxels_.options().kStep = Vec3(options_.kVoxelSize, options_.kVoxelSize, options_.kVoxelSize);
    ref_voxels_.InitializeBuffer();
    return true;
}

bool NdtSolver::BuildRefVoxels(const std::vector<Vec3> &all_ref_p_w) {
    RETURN_FALSE_IF(ref_voxels_.buffer().empty());
    RETURN_FALSE_IF(all_ref_p_w.empty());

    ref_voxels_.ResetBuffer();
    for (const Vec3 &p_w: all_ref_p_w) {
        Voxels<int32_t>::Index voxel_indices;
        CONTINUE_IF(!ref_voxels_.ConvertPositionTo3DofIndices(p_w, voxel_indices));
        const uint32_t index = ref_voxels_.GetBufferIndex(voxel_indices);
        ref_voxels_.GetVoxel(index).pdf.IncrementallyFitDistribution(p_w);
        ref_voxels_.changed_items_indices().insert(index);
    }

    for (const uint32_t &index: ref_voxels_.changed_items_indices()) {
        auto &voxel = ref_voxels_.GetVoxel(index);
        CONTINUE_IF(voxel.pdf.num_of_points() < 5);
        // Normalize covariance by sample size and add small diagonal regularization.
        Mat3 cov = voxel.pdf.covariance();
        const float denom = static_cast<float>(std::max(1u, voxel.pdf.num_of_points() - 1u));
        cov /= denom;
        cov += 1e-4f * Mat3::Identity();
        voxel.inv_cov = cov.inverse();
    }
    return true;
}

bool NdtSolver::EstimatePose(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc) {
    RETURN_FALSE_IF(!BuildRefVoxels(all_ref_p_w));

    q_rc.normalize();
    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();

    const uint32_t index_step = GetIndexStep(all_cur_p_w.size());
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        hessian.setZero();
        bias.setZero();

        // Iterate each current point to construct incremental function.
        for (uint32_t i = 0; i < all_cur_p_w.size(); i += index_step) {
            const Vec3 &cur_p_w = all_cur_p_w[i];
            const Vec3 transformed_cur_p_w = q_rc * cur_p_w + p_rc;

            // Extract normal distribution of reference voxel.
            Voxels<int32_t>::Index voxel_indices;
            CONTINUE_IF(!ref_voxels_.ConvertPositionTo3DofIndices(transformed_cur_p_w, voxel_indices));
            const uint32_t index = ref_voxels_.GetBufferIndex(voxel_indices);
            const auto &voxel = ref_voxels_.GetVoxel(index);
            CONTINUE_IF(voxel.pdf.num_of_points() < 5);

            // Compute residual.
            const Vec3 residual = transformed_cur_p_w - voxel.pdf.mid_point();
            CONTINUE_IF(residual.norm() > options_.kMaxValidRelativePointDistance);

            // Compute jacobian.
            Mat3x6 jacobian = Mat3x6::Zero();
            jacobian.block<3, 3>(0, 0) = Mat3::Identity();
            jacobian.block<3, 3>(0, 3) = -q_rc.toRotationMatrix() * Utility::SkewSymmetricMatrix(cur_p_w);

            // Construct hessian and bias.
            hessian += jacobian.transpose() * voxel.inv_cov * jacobian;
            bias -= jacobian.transpose() * voxel.inv_cov * residual;
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

}  // namespace vision_geometry
