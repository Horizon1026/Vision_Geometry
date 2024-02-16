#ifndef _GEOMETRY_ICP_H_
#define _GEOMETRY_ICP_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

/* Class IcpSolver Declaration. */
class IcpSolver {

public:
    struct IcpOptions {
        uint32_t kMaxIteration = 100;
        float kMaxValidRelativePointDistance = 5.0f;
    };

public:
    explicit IcpSolver() = default;
    virtual ~IcpSolver() = default;

    bool EstimatePose(const std::vector<Vec3> &ref_p_w,
                      const std::vector<Vec3> &cur_p_w,
                      Quat &q_cr,
                      Vec3 &p_cr);
    bool EstimatePoseByPairs(const std::vector<Vec3> &ref_p_w,
                             const std::vector<Vec3> &cur_p_w,
                             Quat &q_cr,
                             Vec3 &p_cr);

    // Reference for member variables.
    IcpOptions &options() { return options_;}

    // Const reference for member variables.
    const IcpOptions &options() const { return options_;}

private:

private:
    IcpOptions options_;
};

}

#endif // end of _GEOMETRY_ICP_H_
