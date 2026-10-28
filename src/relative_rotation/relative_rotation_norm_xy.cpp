#include "relative_rotation.h"

#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace vision_geometry {

bool RelativeRotation::EstimateRotationByBnb(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Quat &q_cr) {
    RETURN_FALSE_IF(ref_norm_xy.empty());
    RETURN_FALSE_IF(ref_norm_xy.size() != cur_norm_xy.size());

    q_cr.normalize();

    SummationTerms terms = {};
    ComputeSummationTerms(ref_norm_xy, cur_norm_xy, terms);
    EstimateRotationByBnb(terms, q_cr);
    return true;
}

bool RelativeRotation::EstimateRotation(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Quat &q_cr) {
    RETURN_FALSE_IF(ref_norm_xy.empty());
    RETURN_FALSE_IF(ref_norm_xy.size() != cur_norm_xy.size());
    q_cr.normalize();

    // Compute summation terms.
    SummationTerms terms = {};
    ComputeSummationTerms(ref_norm_xy, cur_norm_xy, terms);

    // Estimate rotation between reference and current frame.
    EstimateRotationUseAll(terms, q_cr);

    return true;
}

bool RelativeRotation::EstimatePose(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Vec3 &t_cr, Quat &q_cr) {
    RETURN_FALSE_IF(ref_norm_xy.empty());
    RETURN_FALSE_IF(ref_norm_xy.size() != cur_norm_xy.size());
    q_cr.normalize();

    // Compute summation terms.
    SummationTerms terms = {};
    ComputeSummationTerms(ref_norm_xy, cur_norm_xy, terms);

    // Estimate rotation between reference and current frame.
    EstimatePoseUseAll(terms, t_cr, q_cr);

    // Correct the translation.
    const Vec3 f_r = Vec3(ref_norm_xy[0].x(), ref_norm_xy[0].y(), 1.0f);
    const Vec3 f_c = Vec3(cur_norm_xy[0].x(), cur_norm_xy[0].y(), 1.0f);
    const Vec3 temp_f_c = q_cr * f_r;
    const Vec3 optical_flow = f_c - temp_f_c;
    if (optical_flow.dot(t_cr) < 0) {
        t_cr = -t_cr;
    }

    return true;
}

void RelativeRotation::ComputeSummationTerms(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, SummationTerms &terms) {
    for (uint32_t i = 0; i < ref_norm_xy.size(); ++i) {
        const Vec3 f_r = Vec3(ref_norm_xy[i].x(), ref_norm_xy[i].y(), 1.0f);
        const Vec3 f_c = Vec3(cur_norm_xy[i].x(), cur_norm_xy[i].y(), 1.0f);
        AccumulateSummationTerms(f_r, f_c, terms);
    }
}

}  // namespace vision_geometry
