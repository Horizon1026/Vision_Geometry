#ifndef _RELATIVE_ROTATION_H_
#define _RELATIVE_ROTATION_H_

#include "basic_type.h"
#include "eigen_optimization_functor.h"

#include "eigen3/unsupported/Eigen/NonLinearOptimization"
#include "eigen3/unsupported/Eigen/NumericalDiff"

namespace VISION_GEOMETRY {

struct RelativeRotationOptions {
    float kHalfBoundOfPitchInDeg = 20.0f;
    float kHalfBoundOfRollInDeg = 20.0f;
    float kHalfBoundOfYawInDeg = 20.0f;
    int32_t kDivisionsOfPitch = 3;
    int32_t kDivisionsOfRoll = 3;
    int32_t kDivisionsOfYaw = 3;
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

    bool EstimateRotationByBnb(const std::vector<Vec2> &ref_norm_xy,
                               const std::vector<Vec2> &cur_norm_xy,
                               Quat &q_cr);
    bool EstimateRotation(const std::vector<Vec2> &ref_norm_xy,
                          const std::vector<Vec2> &cur_norm_xy,
                          Quat &q_cr);
    bool EstimatePose(const std::vector<Vec2> &ref_norm_xy,
                      const std::vector<Vec2> &cur_norm_xy,
                      Quat &q_cr,
                      Vec3 &t_cr);

    // Reference for member variables.
    RelativeRotationOptions &options() { return options_; }

    // Const reference for member variables.
    const RelativeRotationOptions &options() const { return options_; }

    // Static functinos.
    static void ComputeM(const SummationTerms &terms,
                         const Vec3 &cayley,
                         Mat3 &M);
    static void ComputeMWithJacobians(const SummationTerms &terms,
                                      const Vec3 &cayley,
                                      Jacobians &jacobians,
                                      Mat3 &M);
    static float ComputeSmallestEigenValueAndJacobian(const SummationTerms &terms,
                                                      const Vec3 &cayley,
                                                      Mat1x3 &jacobian);
    static float ComputeSmallestEigenValueWithM(const SummationTerms &terms,
                                                const Vec3 &cayley,
                                                Mat3 &M);
    // Convert feature pairs to summation terms.
    static void ComputeSummationTerms(const std::vector<Vec2> &ref_norm_xy,
                                      const std::vector<Vec2> &cur_norm_xy,
                                      SummationTerms &terms);

private:
    // Only estimate rotation, and return the cayley format of it.
    Vec3 EstimateRotationUseAll(const SummationTerms &terms,
                                Quat &q_cr);
    // Estimate rotation and translation.
    void EstimatePoseUseAll(const SummationTerms &terms,
                            Quat &q_cr,
                            Vec3 &t_cr);

private:
    RelativeRotationOptions options_;
};

/* Eigen Solver Step Definition. */
struct EigenSolverStep : OptimizationFunctor<float> {
    const SummationTerms &terms_;

    EigenSolverStep(const SummationTerms &terms) :
        OptimizationFunctor<float>(3, 3),
        terms_(terms) {}

    int operator()(const Vec &x, Vec &fvec) const {
        Vec3 cayley = x;
        Mat1x3 jacobian = Mat1x3::Zero();

        RelativeRotation::ComputeSmallestEigenValueAndJacobian(terms_, cayley, jacobian);

        fvec[0] = jacobian(0, 0);
        fvec[1] = jacobian(0, 1);
        fvec[2] = jacobian(0, 2);

        return 0;
    }
};

}

#endif // end of _RELATIVE_ROTATION_H_
