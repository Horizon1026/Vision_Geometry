#ifndef _RELATIVE_ROTATION_H_
#define _RELATIVE_ROTATION_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

struct RelativeRotationOptions {
    uint32_t kMaxIteration = 10;
};

struct SummationTerms {
    Mat3 xx = Mat3::Zero();
    Mat3 yy = Mat3::Zero();
    Mat3 zz = Mat3::Zero();
    Mat3 xy = Mat3::Zero();
    Mat3 yz = Mat3::Zero();
    Mat3 zx = Mat3::Zero();
};

struct Jacobians {
    Mat3 jac_1 = Mat3::Zero();
    Mat3 jac_2 = Mat3::Zero();
    Mat3 jac_3 = Mat3::Zero();
};

struct PairOfEigen {
    Vec3 vector = Vec3::Zero();
    float value = 0.0f;
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
    bool EstimateRotationUseAll(const SummationTerms &terms,
                                Quat &q_cr);
    void ComputeMWithJacobians(const SummationTerms &terms,
                               const Vec3 &cayley,
                               Jacobians &jacobians,
                               Mat3 &M);
    float ComputeSmallestEVWithJacobian(const SummationTerms &terms,
                                        const Vec3 &cayley,
                                        Mat1x3 &jacobian);

    void ComputeM(const SummationTerms &terms,
                  const Vec3 &cayley,
                  Mat3 &M);

private:
    RelativeRotationOptions options_;
};
}

#endif // end of _RELATIVE_ROTATION_H_
