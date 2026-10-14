#include "relative_rotation.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

bool RelativeRotation::EstimateRotationByBnb(const SummationTerms &terms, Quat &q_cr) {
    RETURN_FALSE_IF(options_.kDivisionsOfPitch < 2 || options_.kDivisionsOfRoll < 2 || options_.kDivisionsOfYaw < 2);
    RETURN_FALSE_IF(!q_cr.coeffs().allFinite() || q_cr.norm() <= kZeroFloat);
    q_cr.normalize();

    // Compute the step of pitch, roll, yaw.
    const Vec3 step_deg = Vec3(options_.kHalfBoundOfPitchInDeg * 2.0f / static_cast<float>(options_.kDivisionsOfPitch - 1),
                               options_.kHalfBoundOfRollInDeg * 2.0f / static_cast<float>(options_.kDivisionsOfRoll - 1),
                               options_.kHalfBoundOfYawInDeg * 2.0f / static_cast<float>(options_.kDivisionsOfYaw - 1));

    const Vec3 mid_rpy = Utility::QuaternionToEuler(q_cr);

    // Iterate all selected initial value.
    float min_eigen_value = INFINITY;
    Mat3 M = Mat3::Identity();
    for (int32_t i_roll = 0; i_roll < options_.kDivisionsOfRoll; ++i_roll) {
        for (int32_t i_pitch = 0; i_pitch < options_.kDivisionsOfPitch; ++i_pitch) {
            for (int32_t i_yaw = 0; i_yaw < options_.kDivisionsOfYaw; ++i_yaw) {
                const float roll = mid_rpy.x() - options_.kHalfBoundOfRollInDeg + step_deg.y() * i_roll;
                const float pitch = mid_rpy.y() - options_.kHalfBoundOfPitchInDeg + step_deg.x() * i_pitch;
                const float yaw = mid_rpy.z() - options_.kHalfBoundOfYawInDeg + step_deg.z() * i_yaw;
                const Vec3 rpy = Vec3(roll, pitch, yaw);
                Quat temp_q_cr = Utility::EulerToQuaternion(rpy);

                // Estimate rotation between reference and current frame.
                EstimateRotationUseAll(terms, temp_q_cr);
                const Vec3 cayley = Utility::ConvertRotationMatrixToCayley(temp_q_cr.matrix());

                const float eigen_value = ComputeSmallestEigenValueWithM(terms, cayley, M);
                if (eigen_value < min_eigen_value) {
                    min_eigen_value = eigen_value;
                    q_cr = temp_q_cr;
                }
            }
        }
    }

    RETURN_FALSE_IF(!std::isfinite(min_eigen_value));
    q_cr.normalize();

    return q_cr.coeffs().allFinite();
}

void RelativeRotation::AccumulateSummationTerms(const Vec3 &f_r, const Vec3 &f_c, SummationTerms &terms) {
    const Mat3 F = f_r * f_r.transpose();
    terms.xx += f_c.x() * f_c.x() * F;
    terms.yy += f_c.y() * f_c.y() * F;
    terms.zz += f_c.z() * f_c.z() * F;
    terms.xy += f_c.x() * f_c.y() * F;
    terms.yz += f_c.y() * f_c.z() * F;
    terms.zx += f_c.z() * f_c.x() * F;
}

}  // namespace vision_geometry
