#include "relative_rotation.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

#include "limits"

namespace vision_geometry {

void RelativeRotation::ComputeMWithJacobians(const SummationTerms &terms, const Vec3 &cayley, Jacobians &jacobians, Mat3 &M) {
    const Mat3 &R = Utility::ConvertCayleyToReducedRotationMatrix(cayley);

    Mat3 &M_jac1 = jacobians.jac_1;
    Mat3 &M_jac2 = jacobians.jac_2;
    Mat3 &M_jac3 = jacobians.jac_3;

    Mat3 R_jac1 = Mat3::Zero();
    Mat3 R_jac2 = Mat3::Zero();
    Mat3 R_jac3 = Mat3::Zero();

    R_jac1(0, 0) = 2.0f * cayley(0);
    R_jac1(0, 1) = 2.0f * cayley(1);
    R_jac1(0, 2) = 2.0f * cayley(2);
    R_jac1(1, 0) = 2.0f * cayley(1);
    R_jac1(1, 1) = -2.0f * cayley(0);
    R_jac1(1, 2) = -2.0f;
    R_jac1(2, 0) = 2.0f * cayley(2);
    R_jac1(2, 1) = 2.0f;
    R_jac1(2, 2) = -2.0f * cayley(0);
    R_jac2(0, 0) = -2.0f * cayley(1);
    R_jac2(0, 1) = 2.0f * cayley(0);
    R_jac2(0, 2) = 2.0f;
    R_jac2(1, 0) = 2.0f * cayley(0);
    R_jac2(1, 1) = 2.0f * cayley(1);
    R_jac2(1, 2) = 2.0f * cayley(2);
    R_jac2(2, 0) = -2.0f;
    R_jac2(2, 1) = 2.0f * cayley(2);
    R_jac2(2, 2) = -2.0f * cayley(1);
    R_jac3(0, 0) = -2.0f * cayley(2);
    R_jac3(0, 1) = -2.0f;
    R_jac3(0, 2) = 2.0f * cayley(0);
    R_jac3(1, 0) = 2.0f;
    R_jac3(1, 1) = -2.0f * cayley(2);
    R_jac3(1, 2) = 2.0f * cayley(1);
    R_jac3(2, 0) = 2.0f * cayley(0);
    R_jac3(2, 1) = 2.0f * cayley(1);
    R_jac3(2, 2) = 2.0f * cayley(2);

    // Fill the matrix M using the precomputed summation terms. Plus Jacobian.
    M.setZero();
    float temp = R.row(2) * terms.yy * R.row(2).transpose();
    M(0, 0) = temp;
    temp = -2.0 * R.row(2) * terms.yz * R.row(1).transpose();
    M(0, 0) += temp;
    temp = R.row(1) * terms.zz * R.row(1).transpose();
    M(0, 0) += temp;
    temp = 2.0 * R_jac1.row(2) * terms.yy * R.row(2).transpose();
    M_jac1(0, 0) = temp;
    temp = -2.0 * R_jac1.row(2) * terms.yz * R.row(1).transpose();
    M_jac1(0, 0) += temp;
    temp = -2.0 * R.row(2) * terms.yz * R_jac1.row(1).transpose();
    M_jac1(0, 0) += temp;
    temp = 2.0 * R_jac1.row(1) * terms.zz * R.row(1).transpose();
    M_jac1(0, 0) += temp;
    temp = 2.0 * R_jac2.row(2) * terms.yy * R.row(2).transpose();
    M_jac2(0, 0) = temp;
    temp = -2.0 * R_jac2.row(2) * terms.yz * R.row(1).transpose();
    M_jac2(0, 0) += temp;
    temp = -2.0 * R.row(2) * terms.yz * R_jac2.row(1).transpose();
    M_jac2(0, 0) += temp;
    temp = 2.0 * R_jac2.row(1) * terms.zz * R.row(1).transpose();
    M_jac2(0, 0) += temp;
    temp = 2.0 * R_jac3.row(2) * terms.yy * R.row(2).transpose();
    M_jac3(0, 0) = temp;
    temp = -2.0 * R_jac3.row(2) * terms.yz * R.row(1).transpose();
    M_jac3(0, 0) += temp;
    temp = -2.0 * R.row(2) * terms.yz * R_jac3.row(1).transpose();
    M_jac3(0, 0) += temp;
    temp = 2.0 * R_jac3.row(1) * terms.zz * R.row(1).transpose();
    M_jac3(0, 0) += temp;

    temp = R.row(2) * terms.yz * R.row(0).transpose();
    M(0, 1) = temp;
    temp = -1.0 * R.row(2) * terms.xy * R.row(2).transpose();
    M(0, 1) += temp;
    temp = -1.0 * R.row(1) * terms.zz * R.row(0).transpose();
    M(0, 1) += temp;
    temp = R.row(1) * terms.zx * R.row(2).transpose();
    M(0, 1) += temp;
    temp = R_jac1.row(2) * terms.yz * R.row(0).transpose();
    M_jac1(0, 1) = temp;
    temp = R.row(2) * terms.yz * R_jac1.row(0).transpose();
    M_jac1(0, 1) += temp;
    temp = -2.0 * R_jac1.row(2) * terms.xy * R.row(2).transpose();
    M_jac1(0, 1) += temp;
    temp = -R_jac1.row(1) * terms.zz * R.row(0).transpose();
    M_jac1(0, 1) += temp;
    temp = -R.row(1) * terms.zz * R_jac1.row(0).transpose();
    M_jac1(0, 1) += temp;
    temp = R_jac1.row(1) * terms.zx * R.row(2).transpose();
    M_jac1(0, 1) += temp;
    temp = R.row(1) * terms.zx * R_jac1.row(2).transpose();
    M_jac1(0, 1) += temp;
    temp = R_jac2.row(2) * terms.yz * R.row(0).transpose();
    M_jac2(0, 1) = temp;
    temp = R.row(2) * terms.yz * R_jac2.row(0).transpose();
    M_jac2(0, 1) += temp;
    temp = -2.0 * R_jac2.row(2) * terms.xy * R.row(2).transpose();
    M_jac2(0, 1) += temp;
    temp = -R_jac2.row(1) * terms.zz * R.row(0).transpose();
    M_jac2(0, 1) += temp;
    temp = -R.row(1) * terms.zz * R_jac2.row(0).transpose();
    M_jac2(0, 1) += temp;
    temp = R_jac2.row(1) * terms.zx * R.row(2).transpose();
    M_jac2(0, 1) += temp;
    temp = R.row(1) * terms.zx * R_jac2.row(2).transpose();
    M_jac2(0, 1) += temp;
    temp = R_jac3.row(2) * terms.yz * R.row(0).transpose();
    M_jac3(0, 1) = temp;
    temp = R.row(2) * terms.yz * R_jac3.row(0).transpose();
    M_jac3(0, 1) += temp;
    temp = -2.0 * R_jac3.row(2) * terms.xy * R.row(2).transpose();
    M_jac3(0, 1) += temp;
    temp = -R_jac3.row(1) * terms.zz * R.row(0).transpose();
    M_jac3(0, 1) += temp;
    temp = -R.row(1) * terms.zz * R_jac3.row(0).transpose();
    M_jac3(0, 1) += temp;
    temp = R_jac3.row(1) * terms.zx * R.row(2).transpose();
    M_jac3(0, 1) += temp;
    temp = R.row(1) * terms.zx * R_jac3.row(2).transpose();
    M_jac3(0, 1) += temp;

    temp = R.row(2) * terms.xy * R.row(1).transpose();
    M(0, 2) = temp;
    temp = -1.0 * R.row(2) * terms.yy * R.row(0).transpose();
    M(0, 2) += temp;
    temp = -1.0 * R.row(1) * terms.zx * R.row(1).transpose();
    M(0, 2) += temp;
    temp = R.row(1) * terms.yz * R.row(0).transpose();
    M(0, 2) += temp;
    temp = R_jac1.row(2) * terms.xy * R.row(1).transpose();
    M_jac1(0, 2) = temp;
    temp = R.row(2) * terms.xy * R_jac1.row(1).transpose();
    M_jac1(0, 2) += temp;
    temp = -R_jac1.row(2) * terms.yy * R.row(0).transpose();
    M_jac1(0, 2) += temp;
    temp = -R.row(2) * terms.yy * R_jac1.row(0).transpose();
    M_jac1(0, 2) += temp;
    temp = -2.0 * R_jac1.row(1) * terms.zx * R.row(1).transpose();
    M_jac1(0, 2) += temp;
    temp = R_jac1.row(1) * terms.yz * R.row(0).transpose();
    M_jac1(0, 2) += temp;
    temp = R.row(1) * terms.yz * R_jac1.row(0).transpose();
    M_jac1(0, 2) += temp;
    temp = R_jac2.row(2) * terms.xy * R.row(1).transpose();
    M_jac2(0, 2) = temp;
    temp = R.row(2) * terms.xy * R_jac2.row(1).transpose();
    M_jac2(0, 2) += temp;
    temp = -R_jac2.row(2) * terms.yy * R.row(0).transpose();
    M_jac2(0, 2) += temp;
    temp = -R.row(2) * terms.yy * R_jac2.row(0).transpose();
    M_jac2(0, 2) += temp;
    temp = -2.0 * R_jac2.row(1) * terms.zx * R.row(1).transpose();
    M_jac2(0, 2) += temp;
    temp = R_jac2.row(1) * terms.yz * R.row(0).transpose();
    M_jac2(0, 2) += temp;
    temp = R.row(1) * terms.yz * R_jac2.row(0).transpose();
    M_jac2(0, 2) += temp;
    temp = R_jac3.row(2) * terms.xy * R.row(1).transpose();
    M_jac3(0, 2) = temp;
    temp = R.row(2) * terms.xy * R_jac3.row(1).transpose();
    M_jac3(0, 2) += temp;
    temp = -R_jac3.row(2) * terms.yy * R.row(0).transpose();
    M_jac3(0, 2) += temp;
    temp = -R.row(2) * terms.yy * R_jac3.row(0).transpose();
    M_jac3(0, 2) += temp;
    temp = -2.0 * R_jac3.row(1) * terms.zx * R.row(1).transpose();
    M_jac3(0, 2) += temp;
    temp = R_jac3.row(1) * terms.yz * R.row(0).transpose();
    M_jac3(0, 2) += temp;
    temp = R.row(1) * terms.yz * R_jac3.row(0).transpose();
    M_jac3(0, 2) += temp;

    temp = R.row(0) * terms.zz * R.row(0).transpose();
    M(1, 1) = temp;
    temp = -2.0 * R.row(0) * terms.zx * R.row(2).transpose();
    M(1, 1) += temp;
    temp = R.row(2) * terms.xx * R.row(2).transpose();
    M(1, 1) += temp;
    temp = 2.0 * R_jac1.row(0) * terms.zz * R.row(0).transpose();
    M_jac1(1, 1) = temp;
    temp = -2.0 * R_jac1.row(0) * terms.zx * R.row(2).transpose();
    M_jac1(1, 1) += temp;
    temp = -2.0 * R.row(0) * terms.zx * R_jac1.row(2).transpose();
    M_jac1(1, 1) += temp;
    temp = 2.0 * R_jac1.row(2) * terms.xx * R.row(2).transpose();
    M_jac1(1, 1) += temp;
    temp = 2.0 * R_jac2.row(0) * terms.zz * R.row(0).transpose();
    M_jac2(1, 1) = temp;
    temp = -2.0 * R_jac2.row(0) * terms.zx * R.row(2).transpose();
    M_jac2(1, 1) += temp;
    temp = -2.0 * R.row(0) * terms.zx * R_jac2.row(2).transpose();
    M_jac2(1, 1) += temp;
    temp = 2.0 * R_jac2.row(2) * terms.xx * R.row(2).transpose();
    M_jac2(1, 1) += temp;
    temp = 2.0 * R_jac3.row(0) * terms.zz * R.row(0).transpose();
    M_jac3(1, 1) = temp;
    temp = -2.0 * R_jac3.row(0) * terms.zx * R.row(2).transpose();
    M_jac3(1, 1) += temp;
    temp = -2.0 * R.row(0) * terms.zx * R_jac3.row(2).transpose();
    M_jac3(1, 1) += temp;
    temp = 2.0 * R_jac3.row(2) * terms.xx * R.row(2).transpose();
    M_jac3(1, 1) += temp;

    temp = R.row(0) * terms.zx * R.row(1).transpose();
    M(1, 2) = temp;
    temp = -1.0 * R.row(0) * terms.yz * R.row(0).transpose();
    M(1, 2) += temp;
    temp = -1.0 * R.row(2) * terms.xx * R.row(1).transpose();
    M(1, 2) += temp;
    temp = R.row(2) * terms.xy * R.row(0).transpose();
    M(1, 2) += temp;
    temp = R_jac1.row(0) * terms.zx * R.row(1).transpose();
    M_jac1(1, 2) = temp;
    temp = R.row(0) * terms.zx * R_jac1.row(1).transpose();
    M_jac1(1, 2) += temp;
    temp = -2.0 * R_jac1.row(0) * terms.yz * R.row(0).transpose();
    M_jac1(1, 2) += temp;
    temp = -R_jac1.row(2) * terms.xx * R.row(1).transpose();
    M_jac1(1, 2) += temp;
    temp = -R.row(2) * terms.xx * R_jac1.row(1).transpose();
    M_jac1(1, 2) += temp;
    temp = R_jac1.row(2) * terms.xy * R.row(0).transpose();
    M_jac1(1, 2) += temp;
    temp = R.row(2) * terms.xy * R_jac1.row(0).transpose();
    M_jac1(1, 2) += temp;
    temp = R_jac2.row(0) * terms.zx * R.row(1).transpose();
    M_jac2(1, 2) = temp;
    temp = R.row(0) * terms.zx * R_jac2.row(1).transpose();
    M_jac2(1, 2) += temp;
    temp = -2.0 * R_jac2.row(0) * terms.yz * R.row(0).transpose();
    M_jac2(1, 2) += temp;
    temp = -R_jac2.row(2) * terms.xx * R.row(1).transpose();
    M_jac2(1, 2) += temp;
    temp = -R.row(2) * terms.xx * R_jac2.row(1).transpose();
    M_jac2(1, 2) += temp;
    temp = R_jac2.row(2) * terms.xy * R.row(0).transpose();
    M_jac2(1, 2) += temp;
    temp = R.row(2) * terms.xy * R_jac2.row(0).transpose();
    M_jac2(1, 2) += temp;
    temp = R_jac3.row(0) * terms.zx * R.row(1).transpose();
    M_jac3(1, 2) = temp;
    temp = R.row(0) * terms.zx * R_jac3.row(1).transpose();
    M_jac3(1, 2) += temp;
    temp = -2.0 * R_jac3.row(0) * terms.yz * R.row(0).transpose();
    M_jac3(1, 2) += temp;
    temp = -R_jac3.row(2) * terms.xx * R.row(1).transpose();
    M_jac3(1, 2) += temp;
    temp = -R.row(2) * terms.xx * R_jac3.row(1).transpose();
    M_jac3(1, 2) += temp;
    temp = R_jac3.row(2) * terms.xy * R.row(0).transpose();
    M_jac3(1, 2) += temp;
    temp = R.row(2) * terms.xy * R_jac3.row(0).transpose();
    M_jac3(1, 2) += temp;

    temp = R.row(1) * terms.xx * R.row(1).transpose();
    M(2, 2) = temp;
    temp = -2.0 * R.row(0) * terms.xy * R.row(1).transpose();
    M(2, 2) += temp;
    temp = R.row(0) * terms.yy * R.row(0).transpose();
    M(2, 2) += temp;
    temp = 2.0 * R_jac1.row(1) * terms.xx * R.row(1).transpose();
    M_jac1(2, 2) = temp;
    temp = -2.0 * R_jac1.row(0) * terms.xy * R.row(1).transpose();
    M_jac1(2, 2) += temp;
    temp = -2.0 * R.row(0) * terms.xy * R_jac1.row(1).transpose();
    M_jac1(2, 2) += temp;
    temp = 2.0 * R_jac1.row(0) * terms.yy * R.row(0).transpose();
    M_jac1(2, 2) += temp;
    temp = 2.0 * R_jac2.row(1) * terms.xx * R.row(1).transpose();
    M_jac2(2, 2) = temp;
    temp = -2.0 * R_jac2.row(0) * terms.xy * R.row(1).transpose();
    M_jac2(2, 2) += temp;
    temp = -2.0 * R.row(0) * terms.xy * R_jac2.row(1).transpose();
    M_jac2(2, 2) += temp;
    temp = 2.0 * R_jac2.row(0) * terms.yy * R.row(0).transpose();
    M_jac2(2, 2) += temp;
    temp = 2.0 * R_jac3.row(1) * terms.xx * R.row(1).transpose();
    M_jac3(2, 2) = temp;
    temp = -2.0 * R_jac3.row(0) * terms.xy * R.row(1).transpose();
    M_jac3(2, 2) += temp;
    temp = -2.0 * R.row(0) * terms.xy * R_jac3.row(1).transpose();
    M_jac3(2, 2) += temp;
    temp = 2.0 * R_jac3.row(0) * terms.yy * R.row(0).transpose();
    M_jac3(2, 2) += temp;

    M(1, 0) = M(0, 1);
    M(2, 0) = M(0, 2);
    M(2, 1) = M(1, 2);
    M_jac1(1, 0) = M_jac1(0, 1);
    M_jac1(2, 0) = M_jac1(0, 2);
    M_jac1(2, 1) = M_jac1(1, 2);
    M_jac2(1, 0) = M_jac2(0, 1);
    M_jac2(2, 0) = M_jac2(0, 2);
    M_jac2(2, 1) = M_jac2(1, 2);
    M_jac3(1, 0) = M_jac3(0, 1);
    M_jac3(2, 0) = M_jac3(0, 2);
    M_jac3(2, 1) = M_jac3(1, 2);
}

float RelativeRotation::ComputeSmallestEigenValueWithM(const SummationTerms &terms, const Vec3 &cayley, Mat3 &M) {
    ComputeM(terms, cayley, M);

    // Retrieve the smallest Eigenvalue by the following closed form solution
    float b = -M(0, 0) - M(1, 1) - M(2, 2);
    float c = -pow(M(0, 2), 2) - pow(M(1, 2), 2) - pow(M(0, 1), 2) + M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2);
    float d = M(1, 1) * pow(M(0, 2), 2) + M(0, 0) * pow(M(1, 2), 2) + M(2, 2) * pow(M(0, 1), 2) - M(0, 0) * M(1, 1) * M(2, 2) - 2 * M(0, 1) * M(1, 2) * M(0, 2);

    float s = 2.0f * pow(b, 3) - 9.0f * b * c + 27.0f * d;
    float t = 4.0f * pow((pow(b, 2) - 3.0f * c), 3);

    float alpha = acos(s / sqrt(t));
    float beta = alpha / 3.0f;
    float y = cos(beta);

    float r = 0.5f * sqrt(t);
    float w = pow(r, (1.0f / 3.0f));

    float k = w * y;
    float smallestEV = (-b - 2.0f * k) / 3.0f;
    return smallestEV;
}

float RelativeRotation::ComputeSmallestEigenValueAndJacobian(const SummationTerms &terms, const Vec3 &cayley, Mat1x3 &jacobian) {
    Jacobians jacobians;
    Mat3 M = Mat3::Zero();
    ComputeMWithJacobians(terms, cayley, jacobians, M);
    Mat3 &M_jac1 = jacobians.jac_1;
    Mat3 &M_jac2 = jacobians.jac_2;
    Mat3 &M_jac3 = jacobians.jac_3;

    // Retrieve the smallest Eigenvalue by the following closed form solution.
    float b = -M(0, 0) - M(1, 1) - M(2, 2);
    float b_jac1 = -M_jac1(0, 0) - M_jac1(1, 1) - M_jac1(2, 2);
    float b_jac2 = -M_jac2(0, 0) - M_jac2(1, 1) - M_jac2(2, 2);
    float b_jac3 = -M_jac3(0, 0) - M_jac3(1, 1) - M_jac3(2, 2);
    float c = -pow(M(0, 2), 2) - pow(M(1, 2), 2) - pow(M(0, 1), 2) + M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2);
    float c_jac1 = -2.0 * M(0, 2) * M_jac1(0, 2) - 2.0 * M(1, 2) * M_jac1(1, 2) - 2.0 * M(0, 1) * M_jac1(0, 1) + M_jac1(0, 0) * M(1, 1) +
                   M(0, 0) * M_jac1(1, 1) + M_jac1(0, 0) * M(2, 2) + M(0, 0) * M_jac1(2, 2) + M_jac1(1, 1) * M(2, 2) + M(1, 1) * M_jac1(2, 2);
    float c_jac2 = -2.0 * M(0, 2) * M_jac2(0, 2) - 2.0 * M(1, 2) * M_jac2(1, 2) - 2.0 * M(0, 1) * M_jac2(0, 1) + M_jac2(0, 0) * M(1, 1) +
                   M(0, 0) * M_jac2(1, 1) + M_jac2(0, 0) * M(2, 2) + M(0, 0) * M_jac2(2, 2) + M_jac2(1, 1) * M(2, 2) + M(1, 1) * M_jac2(2, 2);
    float c_jac3 = -2.0 * M(0, 2) * M_jac3(0, 2) - 2.0 * M(1, 2) * M_jac3(1, 2) - 2.0 * M(0, 1) * M_jac3(0, 1) + M_jac3(0, 0) * M(1, 1) +
                   M(0, 0) * M_jac3(1, 1) + M_jac3(0, 0) * M(2, 2) + M(0, 0) * M_jac3(2, 2) + M_jac3(1, 1) * M(2, 2) + M(1, 1) * M_jac3(2, 2);
    float d = M(1, 1) * pow(M(0, 2), 2) + M(0, 0) * pow(M(1, 2), 2) + M(2, 2) * pow(M(0, 1), 2) - M(0, 0) * M(1, 1) * M(2, 2) - 2 * M(0, 1) * M(1, 2) * M(0, 2);
    float d_jac1 = M_jac1(1, 1) * pow(M(0, 2), 2) + M(1, 1) * 2 * M(0, 2) * M_jac1(0, 2) + M_jac1(0, 0) * pow(M(1, 2), 2) +
                   M(0, 0) * 2.0 * M(1, 2) * M_jac1(1, 2) + M_jac1(2, 2) * pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac1(0, 1) -
                   M_jac1(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac1(1, 1) * M(2, 2) - M(0, 0) * M(1, 1) * M_jac1(2, 2) -
                   2.0 * (M_jac1(0, 1) * M(1, 2) * M(0, 2) + M(0, 1) * M_jac1(1, 2) * M(0, 2) + M(0, 1) * M(1, 2) * M_jac1(0, 2));
    float d_jac2 = M_jac2(1, 1) * pow(M(0, 2), 2) + M(1, 1) * 2 * M(0, 2) * M_jac2(0, 2) + M_jac2(0, 0) * pow(M(1, 2), 2) +
                   M(0, 0) * 2.0 * M(1, 2) * M_jac2(1, 2) + M_jac2(2, 2) * pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac2(0, 1) -
                   M_jac2(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac2(1, 1) * M(2, 2) - M(0, 0) * M(1, 1) * M_jac2(2, 2) -
                   2.0 * (M_jac2(0, 1) * M(1, 2) * M(0, 2) + M(0, 1) * M_jac2(1, 2) * M(0, 2) + M(0, 1) * M(1, 2) * M_jac2(0, 2));
    float d_jac3 = M_jac3(1, 1) * pow(M(0, 2), 2) + M(1, 1) * 2 * M(0, 2) * M_jac3(0, 2) + M_jac3(0, 0) * pow(M(1, 2), 2) +
                   M(0, 0) * 2.0 * M(1, 2) * M_jac3(1, 2) + M_jac3(2, 2) * pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac3(0, 1) -
                   M_jac3(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac3(1, 1) * M(2, 2) - M(0, 0) * M(1, 1) * M_jac3(2, 2) -
                   2.0 * (M_jac3(0, 1) * M(1, 2) * M(0, 2) + M(0, 1) * M_jac3(1, 2) * M(0, 2) + M(0, 1) * M(1, 2) * M_jac3(0, 2));

    float s = 2 * pow(b, 3) - 9 * b * c + 27 * d;
    float t = 4 * pow((pow(b, 2) - 3 * c), 3);
    float s_jac1 = 2.0 * 3.0 * pow(b, 2) * b_jac1 - 9.0 * b_jac1 * c - 9.0 * b * c_jac1 + 27.0 * d_jac1;
    float s_jac2 = 2.0 * 3.0 * pow(b, 2) * b_jac2 - 9.0 * b_jac2 * c - 9.0 * b * c_jac2 + 27.0 * d_jac2;
    float s_jac3 = 2.0 * 3.0 * pow(b, 2) * b_jac3 - 9.0 * b_jac3 * c - 9.0 * b * c_jac3 + 27.0 * d_jac3;
    float t_jac1 = 4.0 * 3.0 * pow((pow(b, 2) - 3.0 * c), 2) * (2.0 * b * b_jac1 - 3.0 * c_jac1);
    float t_jac2 = 4.0 * 3.0 * pow((pow(b, 2) - 3.0 * c), 2) * (2.0 * b * b_jac2 - 3.0 * c_jac2);
    float t_jac3 = 4.0 * 3.0 * pow((pow(b, 2) - 3.0 * c), 2) * (2.0 * b * b_jac3 - 3.0 * c_jac3);

    float alpha = acos(s / sqrt(t));
    float alpha_jac1 = -1.0 / sqrt(1.0 - (pow(s, 2) / t)) * (s_jac1 * sqrt(t) - s * 0.5 * pow(t, -0.5) * t_jac1) / t;
    float alpha_jac2 = -1.0 / sqrt(1.0 - (pow(s, 2) / t)) * (s_jac2 * sqrt(t) - s * 0.5 * pow(t, -0.5) * t_jac2) / t;
    float alpha_jac3 = -1.0 / sqrt(1.0 - (pow(s, 2) / t)) * (s_jac3 * sqrt(t) - s * 0.5 * pow(t, -0.5) * t_jac3) / t;
    float beta = alpha / 3;
    float beta_jac1 = alpha_jac1 / 3.0;
    float beta_jac2 = alpha_jac2 / 3.0;
    float beta_jac3 = alpha_jac3 / 3.0;
    float y = cos(beta);
    float y_jac1 = -sin(beta) * beta_jac1;
    float y_jac2 = -sin(beta) * beta_jac2;
    float y_jac3 = -sin(beta) * beta_jac3;

    float r = 0.5 * sqrt(t);
    float r_jac1 = 0.25 * pow(t, -0.5) * t_jac1;
    float r_jac2 = 0.25 * pow(t, -0.5) * t_jac2;
    float r_jac3 = 0.25 * pow(t, -0.5) * t_jac3;
    float w = pow(r, (1.0 / 3.0));
    float w_jac1 = (1.0 / 3.0) * pow(r, -2.0 / 3.0) * r_jac1;
    float w_jac2 = (1.0 / 3.0) * pow(r, -2.0 / 3.0) * r_jac2;
    float w_jac3 = (1.0 / 3.0) * pow(r, -2.0 / 3.0) * r_jac3;

    float k = w * y;
    float k_jac1 = w_jac1 * y + w * y_jac1;
    float k_jac2 = w_jac2 * y + w * y_jac2;
    float k_jac3 = w_jac3 * y + w * y_jac3;
    float smallestEV = (-b - 2.0f * k) / 3.0f;
    float smallestEV_jac1 = (-b_jac1 - 2.0f * k_jac1) / 3.0f;
    float smallestEV_jac2 = (-b_jac2 - 2.0f * k_jac2) / 3.0f;
    float smallestEV_jac3 = (-b_jac3 - 2.0f * k_jac3) / 3.0f;

    jacobian << smallestEV_jac1, smallestEV_jac2, smallestEV_jac3;
    return smallestEV;
}

void RelativeRotation::ComputeM(const SummationTerms &terms, const Vec3 &cayley, Mat3 &M) {
    Mat3 R = Utility::ConvertCayleyToReducedRotationMatrix(cayley);

    // Fill the matrix M using the precomputed summation terms.
    float temp = R.row(2) * terms.yy * R.row(2).transpose();
    M(0, 0) = temp;
    temp = -2.0 * R.row(2) * terms.yz * R.row(1).transpose();
    M(0, 0) += temp;
    temp = R.row(1) * terms.zz * R.row(1).transpose();
    M(0, 0) += temp;

    temp = R.row(2) * terms.yz * R.row(0).transpose();
    M(0, 1) = temp;
    temp = -1.0 * R.row(2) * terms.xy * R.row(2).transpose();
    M(0, 1) += temp;
    temp = -1.0 * R.row(1) * terms.zz * R.row(0).transpose();
    M(0, 1) += temp;
    temp = R.row(1) * terms.zx * R.row(2).transpose();
    M(0, 1) += temp;

    temp = R.row(2) * terms.xy * R.row(1).transpose();
    M(0, 2) = temp;
    temp = -1.0 * R.row(2) * terms.yy * R.row(0).transpose();
    M(0, 2) += temp;
    temp = -1.0 * R.row(1) * terms.zx * R.row(1).transpose();
    M(0, 2) += temp;
    temp = R.row(1) * terms.yz * R.row(0).transpose();
    M(0, 2) += temp;

    temp = R.row(0) * terms.zz * R.row(0).transpose();
    M(1, 1) = temp;
    temp = -2.0 * R.row(0) * terms.zx * R.row(2).transpose();
    M(1, 1) += temp;
    temp = R.row(2) * terms.xx * R.row(2).transpose();
    M(1, 1) += temp;

    temp = R.row(0) * terms.zx * R.row(1).transpose();
    M(1, 2) = temp;
    temp = -1.0 * R.row(0) * terms.yz * R.row(0).transpose();
    M(1, 2) += temp;
    temp = -1.0 * R.row(2) * terms.xx * R.row(1).transpose();
    M(1, 2) += temp;
    temp = R.row(2) * terms.xy * R.row(0).transpose();
    M(1, 2) += temp;

    temp = R.row(1) * terms.xx * R.row(1).transpose();
    M(2, 2) = temp;
    temp = -2.0 * R.row(0) * terms.xy * R.row(1).transpose();
    M(2, 2) += temp;
    temp = R.row(0) * terms.yy * R.row(0).transpose();
    M(2, 2) += temp;

    M(1, 0) = M(0, 1);
    M(2, 0) = M(0, 2);
    M(2, 1) = M(1, 2);
}

bool RelativeRotation::EstimateTranslationUseAll(const SummationTerms &terms, const Vec3 &cayley, Vec3 &t_cr) {
    Mat3 M = Mat3::Zero();
    ComputeM(terms, cayley, M);
    // Translation minimizes t^T M t, so use the smallest eigenvalue's eigenvector.
    Eigen::SelfAdjointEigenSolver<Mat3> eig(M);
    RETURN_FALSE_IF(eig.info() != Eigen::Success || !eig.eigenvalues().allFinite());
    RETURN_FALSE_IF(eig.eigenvalues()(1) <= std::numeric_limits<float>::epsilon() * eig.eigenvalues()(2));
    t_cr = eig.eigenvectors().col(0).normalized();

    RETURN_FALSE_IF(!t_cr.allFinite());

    return true;
}

}  // namespace vision_geometry
