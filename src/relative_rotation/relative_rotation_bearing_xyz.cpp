#include "relative_rotation.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

bool RelativeRotation::EstimateRotationByBnb(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Quat &q_cr) {
    RETURN_FALSE_IF(ref_bearing_xyz.empty());
    RETURN_FALSE_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size());

    RETURN_FALSE_IF(options_.kDivisionsOfPitch < 2 || options_.kDivisionsOfRoll < 2 || options_.kDivisionsOfYaw < 2);

    for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        RETURN_FALSE_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                        ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
    }
    RETURN_FALSE_IF(!q_cr.coeffs().allFinite() || q_cr.norm() <= kZeroFloat);
    q_cr.normalize();
    SummationTerms terms = {};
    ComputeSummationTerms(ref_bearing_xyz, cur_bearing_xyz, terms);
    return EstimateRotationByBnb(terms, q_cr);
}

bool RelativeRotation::EstimateRotation(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Quat &q_cr) {
    RETURN_FALSE_IF(ref_bearing_xyz.empty());
    RETURN_FALSE_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size());
    for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        RETURN_FALSE_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                        ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
    }
    RETURN_FALSE_IF(!q_cr.coeffs().allFinite() || q_cr.norm() <= kZeroFloat);
    q_cr.normalize();

    // Compute summation terms.
    SummationTerms terms = {};
    ComputeSummationTerms(ref_bearing_xyz, cur_bearing_xyz, terms);

    // Estimate rotation between reference and current frame.
    EstimateRotationUseAll(terms, q_cr);

    return q_cr.coeffs().allFinite();
}

bool RelativeRotation::EstimatePose(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Vec3 &t_cr, Quat &q_cr) {
    RETURN_FALSE_IF(ref_bearing_xyz.empty());
    RETURN_FALSE_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size());
    for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        RETURN_FALSE_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                        ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
    }
    RETURN_FALSE_IF(!q_cr.coeffs().allFinite() || q_cr.norm() <= kZeroFloat);
    q_cr.normalize();

    // Compute summation terms.
    SummationTerms terms = {};
    ComputeSummationTerms(ref_bearing_xyz, cur_bearing_xyz, terms);

    // Estimate rotation between reference and current frame.
    const Vec3 cayley = EstimateRotationUseAll(terms, q_cr);
    RETURN_FALSE_IF_FALSE(EstimateTranslationUseAll(terms, cayley, t_cr));

    // Positive ranges imply that the unit-ray optical flow agrees with translation.
    float sign_score = 0.0f;
    for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        sign_score += (cur_bearing_xyz[i].normalized() - q_cr * ref_bearing_xyz[i].normalized()).dot(t_cr);
    }
    if (sign_score < 0.0f) {
        t_cr = -t_cr;
    }

    return q_cr.coeffs().allFinite();
}

void RelativeRotation::ComputeSummationTerms(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, SummationTerms &terms) {
    RETURN_IF(ref_bearing_xyz.size() != cur_bearing_xyz.size());
    for (size_t i = 0; i < ref_bearing_xyz.size(); ++i) {
        RETURN_IF(!ref_bearing_xyz[i].allFinite() || !cur_bearing_xyz[i].allFinite() ||
                  ref_bearing_xyz[i].norm() <= kZeroFloat || cur_bearing_xyz[i].norm() <= kZeroFloat);
        const Vec3 f_r = ref_bearing_xyz[i].normalized();
        const Vec3 f_c = cur_bearing_xyz[i].normalized();
        AccumulateSummationTerms(f_r, f_c, terms);
    }
}

}  // namespace vision_geometry
