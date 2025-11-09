#ifndef _GEOMETRY_ICP_H_
#define _GEOMETRY_ICP_H_

#include "basic_type.h"

namespace vision_geometry {

/* Class IcpSolver Declaration. */
class IcpSolver {

public:
    enum class IcpMethod : uint8_t {
        kPointToPoint = 0,
        kPointToLine = 1,
        kPointToPlane = 2,
    };

    struct IcpOptions {
        uint32_t kMaxIteration = 100;
        uint32_t kMaxUsedPoints = 800;
        float kMaxValidRelativePointDistance = 5.0f;
        float kMaxConvergedStepLength = 1e-4f;
        bool kUseNanoFlannKdTree = true;
        IcpMethod kMethod = IcpMethod::kPointToPoint;
    };

public:
    explicit IcpSolver() = default;
    virtual ~IcpSolver() = default;

    bool EstimatePose(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);

    // Reference for member variables.
    IcpOptions &options() { return options_; }

    // Const reference for member variables.
    const IcpOptions &options() const { return options_; }

private:
    // Support for method of point-to-point.
    bool EstimatePoseByMethodPointToPointWithNanoFlann(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);
    bool EstimatePoseByMethodPointToPointWithKdtree(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);
    bool EstimatePoseByPointPairs(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);

    // Support for method of point-to-line.
    bool EstimatePoseByMethodPointToLineWithNanoFlann(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);
    bool EstimatePoseByMethodPointToLineWithKdtree(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);

    // Support for method of point-to-plane.
    bool EstimatePoseByMethodPointToPlaneWithNanoFlann(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);
    bool EstimatePoseByMethodPointToPlaneWithKdtree(const std::vector<Vec3> &all_ref_p_w, const std::vector<Vec3> &all_cur_p_w, Quat &q_rc, Vec3 &p_rc);

    uint32_t GetIndexStep(const uint32_t num_of_points) {
        return std::max(static_cast<uint32_t>(1), static_cast<uint32_t>(num_of_points / options_.kMaxUsedPoints));
    }

private:
    IcpOptions options_;
};

}  // namespace vision_geometry

#endif  // end of _GEOMETRY_ICP_H_
