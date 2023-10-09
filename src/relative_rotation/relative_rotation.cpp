#include "relative_rotation.h"
#include "math_kinematics.h"
#include "slam_operations.h"

namespace VISION_GEOMETRY {

bool RelativeRotation::EstimateRotation(const std::vector<Vec2> &ref_norm_xy,
                                        const std::vector<Vec2> &cur_norm_xy,
                                        Quat &q_cr,
                                        std::vector<uint8_t> &status) {
    // Lift all observations from norm plane to unit sphere.
    std::vector<Vec3> ref_sphere_xyz;
    std::vector<Vec3> cur_sphere_xyz;

    return true;
}

}
