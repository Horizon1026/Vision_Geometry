#ifndef _RELATIVE_ROTATION_H_
#define _RELATIVE_ROTATION_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

struct RelativeRotationOptions {
    uint32_t kMaxIteration = 10;
};

/* Class RelativeRotation Declaration. */
class RelativeRotation {

public:
    RelativeRotation() = default;
    virtual ~RelativeRotation() = default;

    bool EstimateRotation(const std::vector<Vec2> &ref_norm_xy,
                          const std::vector<Vec2> &cur_norm_xy,
                          Quat &q_cr,
                          std::vector<uint8_t> &status);

    // Reference for member variables.
    RelativeRotationOptions &options() { return options_;}

    // Const reference for member variables.
    const RelativeRotationOptions &options() const { return options_;}

private:
    bool EstimateRotationUseAll(const std::vector<Vec2> &ref_norm_xy,
                                const std::vector<Vec2> &cur_norm_xy,
                                Quat &q_cr,
                                std::vector<uint8_t> &status);

private:
    RelativeRotationOptions options_;

};
}

#endif // end of _RELATIVE_ROTATION_H_
