#ifndef _GEOMETRY_ICP_H_
#define _GEOMETRY_ICP_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

/* Class IcpSolver Declaration. */
class IcpSolver {

public:
    enum class IcpMethod: uint8_t {
        kPointToPoint = 0,
        kPointToLine = 1,
        kPointToPlane = 2,
    };

    struct IcpOptions {
        uint32_t kMaxIteration = 100;
        float kMaxValidRelativePointDistance = 5.0f;
        float kMaxConvergedStepLength = 1e-4f;
        IcpMethod kMethod = IcpMethod::kPointToPoint;
    };

public:
    explicit IcpSolver() = default;
    virtual ~IcpSolver() = default;

    bool EstimatePose(const std::vector<Vec3> &all_ref_p_w,
                      const std::vector<Vec3> &all_cur_p_w,
                      Quat &q_rc,
                      Vec3 &p_rc);

    // Reference for member variables.
    IcpOptions &options() { return options_;}

    // Const reference for member variables.
    const IcpOptions &options() const { return options_;}

private:
    // Support for method of point-to-point.
    bool EstimatePoseByMethodPointToPoint(const std::vector<Vec3> &all_ref_p_w,
                                          const std::vector<Vec3> &all_cur_p_w,
                                          Quat &q_rc,
                                          Vec3 &p_rc);
    bool EstimatePoseByPointPairs(const std::vector<Vec3> &all_ref_p_w,
                                  const std::vector<Vec3> &all_cur_p_w,
                                  Quat &q_rc,
                                  Vec3 &p_rc);

    // Support for method of point-to-line.
    bool EstimatePoseByMethodPointToLine(const std::vector<Vec3> &all_ref_p_w,
                                         const std::vector<Vec3> &all_cur_p_w,
                                         Quat &q_rc,
                                         Vec3 &p_rc);

    // Support for method of point-to-plane.
    bool EstimatePoseByMethodPointToPlane(const std::vector<Vec3> &all_ref_p_w,
                                          const std::vector<Vec3> &all_cur_p_w,
                                          Quat &q_rc,
                                          Vec3 &p_rc);

private:
    IcpOptions options_;
};

}

#endif // end of _GEOMETRY_ICP_H_
