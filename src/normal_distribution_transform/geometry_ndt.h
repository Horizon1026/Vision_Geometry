#ifndef _GEOMETRY_NDT_H_
#define _GEOMETRY_NDT_H_

#include "basic_type.h"
#include "basic_voxels.h"
#include "plane.h"

namespace vision_geometry {

/* Class NdtSolver Declaration. */
class NdtSolver {

public:
    struct Options {
        uint32_t kMaxIteration = 100;
        uint32_t kMaxUsedPoints = 800;
        float kMaxLidarScanRadius = 30.0f;
        float kVoxelSize = 0.5f;
        float kMaxValidRelativePointDistance = 5.0f;
        float kMaxConvergedStepLength = 1e-4f;
    };
    struct Voxel {
        Plane3D plane;
        Mat3 inv_cov = Mat3::Zero();

        bool operator==(const Voxel &rhs) const {
            return plane == rhs.plane && inv_cov == rhs.inv_cov;
        }
        bool operator!=(const Voxel &rhs) const { return !(*this == rhs); }
    };

public:
    NdtSolver();
    virtual ~NdtSolver() = default;

    bool EstimatePose(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    bool BuildRefVoxels(const std::vector<Vec3> &all_ref_p_w);
    uint32_t GetIndexStep(const uint32_t num_of_points) {
        return std::max(static_cast<uint32_t>(1), static_cast<uint32_t>(num_of_points / options_.kMaxUsedPoints));
    }

private:
    Options options_;
    BasicVoxels<Voxel> ref_voxels_;
};

}  // namespace vision_geometry

#endif  // end of _GEOMETRY_NDT_H_
