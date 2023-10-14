#include "relative_rotation.h"
#include "math_kinematics.h"
#include "slam_operations.h"
#include "log_report.h"

namespace VISION_GEOMETRY {

bool RelativeRotation::EstimateRotation(const std::vector<Vec2> &ref_norm_xy,
                                        const std::vector<Vec2> &cur_norm_xy,
                                        Quat &q_cr,
                                        std::vector<uint8_t> &status) {
    RETURN_FALSE_IF(ref_norm_xy.size() != cur_norm_xy.size());

    // Lift all observations from norm plane to unit sphere.
    std::vector<Vec3> ref_sphere_xyz;
    std::vector<Vec3> cur_sphere_xyz;
    ref_sphere_xyz.reserve(ref_norm_xy.size());
    cur_sphere_xyz.reserve(cur_norm_xy.size());
    for (const auto &norm_xy : ref_norm_xy) {
        ref_sphere_xyz.emplace_back(Vec3(norm_xy.x(), norm_xy.y(), 1.0f).normalized());
    }
    for (const auto &norm_xy : cur_norm_xy) {
        cur_sphere_xyz.emplace_back(Vec3(norm_xy.x(), norm_xy.y(), 1.0f).normalized());
    }

    // Compute summation terms.
    SummationTerms terms;
    for (uint32_t i = 0; i < ref_sphere_xyz.size(); ++i) {
        const Vec3 &f1 = ref_sphere_xyz[i];
        const Vec3 &f2 = cur_sphere_xyz[i];
        const Mat3 F = f2 * f2.transpose();
        const float weight = 1.0f;

        terms.xx += weight * f1.x() * f1.x() * F;
        terms.yy += weight * f1.y() * f1.y() * F;
        terms.zz += weight * f1.z() * f1.z() * F;
        terms.xy += weight * f1.x() * f1.y() * F;
        terms.yz += weight * f1.y() * f1.z() * F;
        terms.zx += weight * f1.z() * f1.x() * F;
    }

    // Estimate rotation between reference and current frame.
    if (!EstimateRotationUseAll(terms, q_cr)) {
        ReportError("[Relative Rotation] Failed to estimate rotation.");
        return false;
    }
    return true;
}

bool RelativeRotation::EstimateRotationUseAll(const SummationTerms &terms,
                                              Quat &q_cr) {
    // Prepare for optimizaiton.
    Vec3 cayley = Utility::ConvertRotationMatrixToCayley(q_cr.matrix());

    // Optimize cayley.
    // TODO:

    // Compute matrix M.
    const Mat3 R_cr = Utility::ConvertCayleyToRotationMatrix(cayley);
    q_cr = Quat(R_cr);
    Mat3 M = Mat3::Zero();
    ComputeM(terms, cayley, M);

    // Decompose matrix M.
    Eigen::EigenSolver<Mat3> eig(M, true);
    Eigen::Matrix<std::complex<float>, 3, 1> D_complex = eig.eigenvalues();
    Eigen::Matrix<std::complex<float>, 3, 3> V_complex = eig.eigenvectors();

    // Sort eigen vectors with values.
    std::map<float, Vec3> eigen_values_vectors;
    for (uint32_t i = 0; i < 3; ++i) {
        Vec3 eigen_vector;
        for (uint32_t j = 0; j < 3; ++j) {
            eigen_vector(j) = V_complex(j, i).real();
        }
        eigen_values_vectors.insert(std::make_pair(D_complex[i].real(), eigen_vector));
    }

    // Record sorted eigen vectors and values.
    Vec3 eigen_values = Vec3::Zero();
    Mat3 eigen_vectors = Mat3::Zero();
    uint32_t idx = 0;
    for (const auto &eigen_value_vector : eigen_values_vectors) {
        eigen_values(idx) = eigen_value_vector.first;
        eigen_vectors.col(idx) = eigen_value_vector.second;
    }

    // Compute translation.
    Vec3 t_cr = eigen_values.head<2>().norm() * eigen_vectors.col(2);

    return true;
}

void RelativeRotation::ComputeMWithJacobians(const SummationTerms &terms,
                                             const Vec3 &cayley,
                                             Jacobians &jacobians,
                                             Mat3 &M) {
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

    //Fill the matrix M using the precomputed summation terms. Plus Jacobian.
    M.setZero();
    float temp = R.row(2)*terms.yy*R.row(2).transpose();
    M(0,0) = temp;
    temp = -2.0*R.row(2)*terms.yz*R.row(1).transpose();
    M(0,0) += temp;
    temp = R.row(1)*terms.zz*R.row(1).transpose();
    M(0,0) += temp;
    temp = 2.0*R_jac1.row(2)*terms.yy*R.row(2).transpose();
    M_jac1(0,0)  = temp;
    temp = -2.0*R_jac1.row(2)*terms.yz*R.row(1).transpose();
    M_jac1(0,0) += temp;
    temp = -2.0*R.row(2)*terms.yz*R_jac1.row(1).transpose();
    M_jac1(0,0) += temp;
    temp = 2.0*R_jac1.row(1)*terms.zz*R.row(1).transpose();
    M_jac1(0,0) += temp;
    temp = 2.0*R_jac2.row(2)*terms.yy*R.row(2).transpose();
    M_jac2(0,0)  = temp;
    temp = -2.0*R_jac2.row(2)*terms.yz*R.row(1).transpose();
    M_jac2(0,0) += temp;
    temp = -2.0*R.row(2)*terms.yz*R_jac2.row(1).transpose();
    M_jac2(0,0) += temp;
    temp = 2.0*R_jac2.row(1)*terms.zz*R.row(1).transpose();
    M_jac2(0,0) += temp;
    temp = 2.0*R_jac3.row(2)*terms.yy*R.row(2).transpose();
    M_jac3(0,0)  = temp;
    temp = -2.0*R_jac3.row(2)*terms.yz*R.row(1).transpose();
    M_jac3(0,0) += temp;
    temp = -2.0*R.row(2)*terms.yz*R_jac3.row(1).transpose();
    M_jac3(0,0) += temp;
    temp = 2.0*R_jac3.row(1)*terms.zz*R.row(1).transpose();
    M_jac3(0,0) += temp;

    temp =      R.row(2)*terms.yz*R.row(0).transpose();
    M(0,1)  = temp;
    temp = -1.0*R.row(2)*terms.xy*R.row(2).transpose();
    M(0,1) += temp;
    temp = -1.0*R.row(1)*terms.zz*R.row(0).transpose();
    M(0,1) += temp;
    temp =      R.row(1)*terms.zx*R.row(2).transpose();
    M(0,1) += temp;
    temp = R_jac1.row(2)*terms.yz*R.row(0).transpose();
    M_jac1(0,1)  = temp;
    temp = R.row(2)*terms.yz*R_jac1.row(0).transpose();
    M_jac1(0,1) += temp;
    temp = -2.0*R_jac1.row(2)*terms.xy*R.row(2).transpose();
    M_jac1(0,1) += temp;
    temp = -R_jac1.row(1)*terms.zz*R.row(0).transpose();
    M_jac1(0,1) += temp;
    temp = -R.row(1)*terms.zz*R_jac1.row(0).transpose();
    M_jac1(0,1) += temp;
    temp = R_jac1.row(1)*terms.zx*R.row(2).transpose();
    M_jac1(0,1) += temp;
    temp = R.row(1)*terms.zx*R_jac1.row(2).transpose();
    M_jac1(0,1) += temp;
    temp = R_jac2.row(2)*terms.yz*R.row(0).transpose();
    M_jac2(0,1)  = temp;
    temp = R.row(2)*terms.yz*R_jac2.row(0).transpose();
    M_jac2(0,1) += temp;
    temp = -2.0*R_jac2.row(2)*terms.xy*R.row(2).transpose();
    M_jac2(0,1) += temp;
    temp = -R_jac2.row(1)*terms.zz*R.row(0).transpose();
    M_jac2(0,1) += temp;
    temp = -R.row(1)*terms.zz*R_jac2.row(0).transpose();
    M_jac2(0,1) += temp;
    temp = R_jac2.row(1)*terms.zx*R.row(2).transpose();
    M_jac2(0,1) += temp;
    temp = R.row(1)*terms.zx*R_jac2.row(2).transpose();
    M_jac2(0,1) += temp;
    temp = R_jac3.row(2)*terms.yz*R.row(0).transpose();
    M_jac3(0,1)  = temp;
    temp = R.row(2)*terms.yz*R_jac3.row(0).transpose();
    M_jac3(0,1) += temp;
    temp = -2.0*R_jac3.row(2)*terms.xy*R.row(2).transpose();
    M_jac3(0,1) += temp;
    temp = -R_jac3.row(1)*terms.zz*R.row(0).transpose();
    M_jac3(0,1) += temp;
    temp = -R.row(1)*terms.zz*R_jac3.row(0).transpose();
    M_jac3(0,1) += temp;
    temp = R_jac3.row(1)*terms.zx*R.row(2).transpose();
    M_jac3(0,1) += temp;
    temp = R.row(1)*terms.zx*R_jac3.row(2).transpose();
    M_jac3(0,1) += temp;

    temp =      R.row(2)*terms.xy*R.row(1).transpose();
    M(0,2)  = temp;
    temp = -1.0*R.row(2)*terms.yy*R.row(0).transpose();
    M(0,2) += temp;
    temp = -1.0*R.row(1)*terms.zx*R.row(1).transpose();
    M(0,2) += temp;
    temp =      R.row(1)*terms.yz*R.row(0).transpose();
    M(0,2) += temp;
    temp = R_jac1.row(2)*terms.xy*R.row(1).transpose();
    M_jac1(0,2)  = temp;
    temp = R.row(2)*terms.xy*R_jac1.row(1).transpose();
    M_jac1(0,2) += temp;
    temp = -R_jac1.row(2)*terms.yy*R.row(0).transpose();
    M_jac1(0,2) += temp;
    temp = -R.row(2)*terms.yy*R_jac1.row(0).transpose();
    M_jac1(0,2) += temp;
    temp = -2.0*R_jac1.row(1)*terms.zx*R.row(1).transpose();
    M_jac1(0,2) += temp;
    temp = R_jac1.row(1)*terms.yz*R.row(0).transpose();
    M_jac1(0,2) += temp;
    temp = R.row(1)*terms.yz*R_jac1.row(0).transpose();
    M_jac1(0,2) += temp;
    temp = R_jac2.row(2)*terms.xy*R.row(1).transpose();
    M_jac2(0,2)  = temp;
    temp = R.row(2)*terms.xy*R_jac2.row(1).transpose();
    M_jac2(0,2) += temp;
    temp = -R_jac2.row(2)*terms.yy*R.row(0).transpose();
    M_jac2(0,2) += temp;
    temp = -R.row(2)*terms.yy*R_jac2.row(0).transpose();
    M_jac2(0,2) += temp;
    temp = -2.0*R_jac2.row(1)*terms.zx*R.row(1).transpose();
    M_jac2(0,2) += temp;
    temp = R_jac2.row(1)*terms.yz*R.row(0).transpose();
    M_jac2(0,2) += temp;
    temp = R.row(1)*terms.yz*R_jac2.row(0).transpose();
    M_jac2(0,2) += temp;
    temp = R_jac3.row(2)*terms.xy*R.row(1).transpose();
    M_jac3(0,2)  = temp;
    temp = R.row(2)*terms.xy*R_jac3.row(1).transpose();
    M_jac3(0,2) += temp;
    temp = -R_jac3.row(2)*terms.yy*R.row(0).transpose();
    M_jac3(0,2) += temp;
    temp = -R.row(2)*terms.yy*R_jac3.row(0).transpose();
    M_jac3(0,2) += temp;
    temp = -2.0*R_jac3.row(1)*terms.zx*R.row(1).transpose();
    M_jac3(0,2) += temp;
    temp = R_jac3.row(1)*terms.yz*R.row(0).transpose();
    M_jac3(0,2) += temp;
    temp = R.row(1)*terms.yz*R_jac3.row(0).transpose();
    M_jac3(0,2) += temp;

    temp =      R.row(0)*terms.zz*R.row(0).transpose();
    M(1,1)  = temp;
    temp = -2.0*R.row(0)*terms.zx*R.row(2).transpose();
    M(1,1) += temp;
    temp =      R.row(2)*terms.xx*R.row(2).transpose();
    M(1,1) += temp;
    temp = 2.0*R_jac1.row(0)*terms.zz*R.row(0).transpose();
    M_jac1(1,1)  = temp;
    temp = -2.0*R_jac1.row(0)*terms.zx*R.row(2).transpose();
    M_jac1(1,1) += temp;
    temp = -2.0*R.row(0)*terms.zx*R_jac1.row(2).transpose();
    M_jac1(1,1) += temp;
    temp = 2.0*R_jac1.row(2)*terms.xx*R.row(2).transpose();
    M_jac1(1,1) += temp;
    temp = 2.0*R_jac2.row(0)*terms.zz*R.row(0).transpose();
    M_jac2(1,1)  = temp;
    temp = -2.0*R_jac2.row(0)*terms.zx*R.row(2).transpose();
    M_jac2(1,1) += temp;
    temp = -2.0*R.row(0)*terms.zx*R_jac2.row(2).transpose();
    M_jac2(1,1) += temp;
    temp = 2.0*R_jac2.row(2)*terms.xx*R.row(2).transpose();
    M_jac2(1,1) += temp;
    temp = 2.0*R_jac3.row(0)*terms.zz*R.row(0).transpose();
    M_jac3(1,1)  = temp;
    temp = -2.0*R_jac3.row(0)*terms.zx*R.row(2).transpose();
    M_jac3(1,1) += temp;
    temp = -2.0*R.row(0)*terms.zx*R_jac3.row(2).transpose();
    M_jac3(1,1) += temp;
    temp = 2.0*R_jac3.row(2)*terms.xx*R.row(2).transpose();
    M_jac3(1,1) += temp;

    temp =      R.row(0)*terms.zx*R.row(1).transpose();
    M(1,2)  = temp;
    temp = -1.0*R.row(0)*terms.yz*R.row(0).transpose();
    M(1,2) += temp;
    temp = -1.0*R.row(2)*terms.xx*R.row(1).transpose();
    M(1,2) += temp;
    temp =      R.row(2)*terms.xy*R.row(0).transpose();
    M(1,2) += temp;
    temp = R_jac1.row(0)*terms.zx*R.row(1).transpose();
    M_jac1(1,2)  = temp;
    temp = R.row(0)*terms.zx*R_jac1.row(1).transpose();
    M_jac1(1,2) += temp;
    temp = -2.0*R_jac1.row(0)*terms.yz*R.row(0).transpose();
    M_jac1(1,2) += temp;
    temp = -R_jac1.row(2)*terms.xx*R.row(1).transpose();
    M_jac1(1,2) += temp;
    temp = -R.row(2)*terms.xx*R_jac1.row(1).transpose();
    M_jac1(1,2) += temp;
    temp = R_jac1.row(2)*terms.xy*R.row(0).transpose();
    M_jac1(1,2) += temp;
    temp = R.row(2)*terms.xy*R_jac1.row(0).transpose();
    M_jac1(1,2) += temp;
    temp = R_jac2.row(0)*terms.zx*R.row(1).transpose();
    M_jac2(1,2)  = temp;
    temp = R.row(0)*terms.zx*R_jac2.row(1).transpose();
    M_jac2(1,2) += temp;
    temp = -2.0*R_jac2.row(0)*terms.yz*R.row(0).transpose();
    M_jac2(1,2) += temp;
    temp = -R_jac2.row(2)*terms.xx*R.row(1).transpose();
    M_jac2(1,2) += temp;
    temp = -R.row(2)*terms.xx*R_jac2.row(1).transpose();
    M_jac2(1,2) += temp;
    temp = R_jac2.row(2)*terms.xy*R.row(0).transpose();
    M_jac2(1,2) += temp;
    temp = R.row(2)*terms.xy*R_jac2.row(0).transpose();
    M_jac2(1,2) += temp;
    temp = R_jac3.row(0)*terms.zx*R.row(1).transpose();
    M_jac3(1,2)  = temp;
    temp = R.row(0)*terms.zx*R_jac3.row(1).transpose();
    M_jac3(1,2) += temp;
    temp = -2.0*R_jac3.row(0)*terms.yz*R.row(0).transpose();
    M_jac3(1,2) += temp;
    temp = -R_jac3.row(2)*terms.xx*R.row(1).transpose();
    M_jac3(1,2) += temp;
    temp = -R.row(2)*terms.xx*R_jac3.row(1).transpose();
    M_jac3(1,2) += temp;
    temp = R_jac3.row(2)*terms.xy*R.row(0).transpose();
    M_jac3(1,2) += temp;
    temp = R.row(2)*terms.xy*R_jac3.row(0).transpose();
    M_jac3(1,2) += temp;

    temp =      R.row(1)*terms.xx*R.row(1).transpose();
    M(2,2)  = temp;
    temp = -2.0*R.row(0)*terms.xy*R.row(1).transpose();
    M(2,2) += temp;
    temp =      R.row(0)*terms.yy*R.row(0).transpose();
    M(2,2) += temp;
    temp = 2.0*R_jac1.row(1)*terms.xx*R.row(1).transpose();
    M_jac1(2,2)  = temp;
    temp = -2.0*R_jac1.row(0)*terms.xy*R.row(1).transpose();
    M_jac1(2,2) += temp;
    temp = -2.0*R.row(0)*terms.xy*R_jac1.row(1).transpose();
    M_jac1(2,2) += temp;
    temp = 2.0*R_jac1.row(0)*terms.yy*R.row(0).transpose();
    M_jac1(2,2) += temp;
    temp = 2.0*R_jac2.row(1)*terms.xx*R.row(1).transpose();
    M_jac2(2,2)  = temp;
    temp = -2.0*R_jac2.row(0)*terms.xy*R.row(1).transpose();
    M_jac2(2,2) += temp;
    temp = -2.0*R.row(0)*terms.xy*R_jac2.row(1).transpose();
    M_jac2(2,2) += temp;
    temp = 2.0*R_jac2.row(0)*terms.yy*R.row(0).transpose();
    M_jac2(2,2) += temp;
    temp = 2.0*R_jac3.row(1)*terms.xx*R.row(1).transpose();
    M_jac3(2,2)  = temp;
    temp = -2.0*R_jac3.row(0)*terms.xy*R.row(1).transpose();
    M_jac3(2,2) += temp;
    temp = -2.0*R.row(0)*terms.xy*R_jac3.row(1).transpose();
    M_jac3(2,2) += temp;
    temp = 2.0*R_jac3.row(0)*terms.yy*R.row(0).transpose();
    M_jac3(2,2) += temp;

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

float RelativeRotation::ComputeSmallestEVWithJacobian(const SummationTerms &terms,
                                                      const Vec3 &cayley,
                                                      Mat1x3 &jacobian) {
    Jacobians jacobians;
    Mat3 M = Mat3::Zero();
    ComputeMWithJacobians(terms, cayley, jacobians, M);
    Mat3 &M_jac1 = jacobians.jac_1;
    Mat3 &M_jac2 = jacobians.jac_2;
    Mat3 &M_jac3 = jacobians.jac_3;

    // Retrieve the smallest Eigenvalue by the following closed form solution.
    // Plus Jacobian.
    float b = - M(0, 0) - M(1, 1) - M(2, 2);
    float b_jac1 = - M_jac1(0, 0) - M_jac1(1, 1) - M_jac1(2, 2);
    float b_jac2 = - M_jac2(0, 0) - M_jac2(1, 1) - M_jac2(2, 2);
    float b_jac3 = - M_jac3(0, 0) - M_jac3(1, 1) - M_jac3(2, 2);
    float c = - pow(M(0, 2), 2) - pow(M(1, 2), 2) - pow(M(0, 1), 2) +
        M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2);
    float c_jac1 = -2.0*M(0,2)*M_jac1(0,2)-2.0*M(1,2)*M_jac1(1,2)-2.0*M(0,1)*M_jac1(0,1)
        +M_jac1(0,0)*M(1,1)+M(0,0)*M_jac1(1,1)+M_jac1(0,0)*M(2,2)
        +M(0,0)*M_jac1(2,2)+M_jac1(1,1)*M(2,2)+M(1,1)*M_jac1(2,2);
    float c_jac2 =
        -2.0*M(0,2)*M_jac2(0,2)-2.0*M(1,2)*M_jac2(1,2)-2.0*M(0,1)*M_jac2(0,1)
        +M_jac2(0,0)*M(1,1)+M(0,0)*M_jac2(1,1)+M_jac2(0,0)*M(2,2)
        +M(0,0)*M_jac2(2,2)+M_jac2(1,1)*M(2,2)+M(1,1)*M_jac2(2,2);
    float c_jac3 =
        -2.0*M(0,2)*M_jac3(0,2)-2.0*M(1,2)*M_jac3(1,2)-2.0*M(0,1)*M_jac3(0,1)
        +M_jac3(0,0)*M(1,1)+M(0,0)*M_jac3(1,1)+M_jac3(0,0)*M(2,2)
        +M(0,0)*M_jac3(2,2)+M_jac3(1,1)*M(2,2)+M(1,1)*M_jac3(2,2);
    float d =
        M(1,1)*pow(M(0,2),2)+M(0,0)*pow(M(1,2),2)+M(2,2)*pow(M(0,1),2)-
        M(0,0)*M(1,1)*M(2,2)-2*M(0,1)*M(1,2)*M(0,2);
    float d_jac1 =
        M_jac1(1,1)*pow(M(0,2),2)+M(1,1)*2*M(0,2)*M_jac1(0,2)
        +M_jac1(0,0)*pow(M(1,2),2)+M(0,0)*2.0*M(1,2)*M_jac1(1,2)
        +M_jac1(2,2)*pow(M(0,1),2)+M(2,2)*2.0*M(0,1)*M_jac1(0,1)
        -M_jac1(0,0)*M(1,1)*M(2,2)-M(0,0)*M_jac1(1,1)*M(2,2)
        -M(0,0)*M(1,1)*M_jac1(2,2)-2.0*(M_jac1(0,1)*M(1,2)*M(0,2)
        +M(0,1)*M_jac1(1,2)*M(0,2)+M(0,1)*M(1,2)*M_jac1(0,2));
    float d_jac2 =
        M_jac2(1,1)*pow(M(0,2),2)+M(1,1)*2*M(0,2)*M_jac2(0,2)
        +M_jac2(0,0)*pow(M(1,2),2)+M(0,0)*2.0*M(1,2)*M_jac2(1,2)
        +M_jac2(2,2)*pow(M(0,1),2)+M(2,2)*2.0*M(0,1)*M_jac2(0,1)
        -M_jac2(0,0)*M(1,1)*M(2,2)-M(0,0)*M_jac2(1,1)*M(2,2)
        -M(0,0)*M(1,1)*M_jac2(2,2)-2.0*(M_jac2(0,1)*M(1,2)*M(0,2)
        +M(0,1)*M_jac2(1,2)*M(0,2)+M(0,1)*M(1,2)*M_jac2(0,2));
    float d_jac3 =
        M_jac3(1,1)*pow(M(0,2),2)+M(1,1)*2*M(0,2)*M_jac3(0,2)
        +M_jac3(0,0)*pow(M(1,2),2)+M(0,0)*2.0*M(1,2)*M_jac3(1,2)
        +M_jac3(2,2)*pow(M(0,1),2)+M(2,2)*2.0*M(0,1)*M_jac3(0,1)
        -M_jac3(0,0)*M(1,1)*M(2,2)-M(0,0)*M_jac3(1,1)*M(2,2)
        -M(0,0)*M(1,1)*M_jac3(2,2)-2.0*(M_jac3(0,1)*M(1,2)*M(0,2)
        +M(0,1)*M_jac3(1,2)*M(0,2)+M(0,1)*M(1,2)*M_jac3(0,2));

    float s = 2*pow(b,3)-9*b*c+27*d;
    float t = 4*pow((pow(b,2)-3*c),3);
    float s_jac1 = 2.0*3.0*pow(b,2)*b_jac1-9.0*b_jac1*c-9.0*b*c_jac1+27.0*d_jac1;
    float s_jac2 = 2.0*3.0*pow(b,2)*b_jac2-9.0*b_jac2*c-9.0*b*c_jac2+27.0*d_jac2;
    float s_jac3 = 2.0*3.0*pow(b,2)*b_jac3-9.0*b_jac3*c-9.0*b*c_jac3+27.0*d_jac3;
    float t_jac1 = 4.0*3.0*pow((pow(b,2)-3.0*c),2)*(2.0*b*b_jac1-3.0*c_jac1);
    float t_jac2 = 4.0*3.0*pow((pow(b,2)-3.0*c),2)*(2.0*b*b_jac2-3.0*c_jac2);
    float t_jac3 = 4.0*3.0*pow((pow(b,2)-3.0*c),2)*(2.0*b*b_jac3-3.0*c_jac3);

    float alpha = acos(s/sqrt(t));
    float alpha_jac1 =
        -1.0/sqrt(1.0-(pow(s,2)/t)) *
        (s_jac1*sqrt(t)-s*0.5*pow(t,-0.5)*t_jac1)/t;
    float alpha_jac2 =
        -1.0/sqrt(1.0-(pow(s,2)/t)) *
        (s_jac2*sqrt(t)-s*0.5*pow(t,-0.5)*t_jac2)/t;
    float alpha_jac3 =
        -1.0/sqrt(1.0-(pow(s,2)/t)) *
        (s_jac3*sqrt(t)-s*0.5*pow(t,-0.5)*t_jac3)/t;
    float beta = alpha/3;
    float beta_jac1 = alpha_jac1/3.0;
    float beta_jac2 = alpha_jac2/3.0;
    float beta_jac3 = alpha_jac3/3.0;
    float y = cos(beta);
    float y_jac1 = -sin(beta)*beta_jac1;
    float y_jac2 = -sin(beta)*beta_jac2;
    float y_jac3 = -sin(beta)*beta_jac3;

    float r = 0.5*sqrt(t);
    float r_jac1 = 0.25*pow(t,-0.5)*t_jac1;
    float r_jac2 = 0.25*pow(t,-0.5)*t_jac2;
    float r_jac3 = 0.25*pow(t,-0.5)*t_jac3;
    float w = pow(r,(1.0/3.0));
    float w_jac1 = (1.0/3.0)*pow(r,-2.0/3.0)*r_jac1;
    float w_jac2 = (1.0/3.0)*pow(r,-2.0/3.0)*r_jac2;
    float w_jac3 = (1.0/3.0)*pow(r,-2.0/3.0)*r_jac3;

    float k = w * y;
    float k_jac1 = w_jac1 * y + w * y_jac1;
    float k_jac2 = w_jac2 * y + w * y_jac2;
    float k_jac3 = w_jac3 * y + w * y_jac3;
    float smallestEV = (- b - 2.0f * k) / 3.0f;
    float smallestEV_jac1 = (- b_jac1 - 2.0f * k_jac1) / 3.0f;
    float smallestEV_jac2 = (- b_jac2 - 2.0f * k_jac2) / 3.0f;
    float smallestEV_jac3 = (- b_jac3 - 2.0f * k_jac3) / 3.0f;

    jacobian << smallestEV_jac1,
                smallestEV_jac2,
                smallestEV_jac3;
    return smallestEV;
}

void RelativeRotation::ComputeM(const SummationTerms &terms,
                                const Vec3 &cayley,
                                Mat3 &M) {
    Mat3 R = Utility::ConvertCayleyToReducedRotationMatrix(cayley);

    // Fill the matrix M using the precomputed summation terms.
    float temp = R.row(2)*terms.yy*R.row(2).transpose();
    M(0,0) = temp;
    temp = -2.0*R.row(2)*terms.yz*R.row(1).transpose();
    M(0,0) += temp;
    temp = R.row(1)*terms.zz*R.row(1).transpose();
    M(0,0) += temp;

    temp = R.row(2)*terms.yz*R.row(0).transpose();
    M(0,1) = temp;
    temp = -1.0*R.row(2)*terms.xy*R.row(2).transpose();
    M(0,1) += temp;
    temp = -1.0*R.row(1)*terms.zz*R.row(0).transpose();
    M(0,1) += temp;
    temp = R.row(1)*terms.zx*R.row(2).transpose();
    M(0,1) += temp;

    temp = R.row(2)*terms.xy*R.row(1).transpose();
    M(0,2) = temp;
    temp = -1.0*R.row(2)*terms.yy*R.row(0).transpose();
    M(0,2) += temp;
    temp = -1.0*R.row(1)*terms.zx*R.row(1).transpose();
    M(0,2) += temp;
    temp = R.row(1)*terms.yz*R.row(0).transpose();
    M(0,2) += temp;

    temp = R.row(0)*terms.zz*R.row(0).transpose();
    M(1,1) = temp;
    temp = -2.0*R.row(0)*terms.zx*R.row(2).transpose();
    M(1,1) += temp;
    temp = R.row(2)*terms.xx*R.row(2).transpose();
    M(1,1) += temp;

    temp = R.row(0)*terms.zx*R.row(1).transpose();
    M(1,2)  = temp;
    temp = -1.0*R.row(0)*terms.yz*R.row(0).transpose();
    M(1,2) += temp;
    temp = -1.0*R.row(2)*terms.xx*R.row(1).transpose();
    M(1,2) += temp;
    temp = R.row(2)*terms.xy*R.row(0).transpose();
    M(1,2) += temp;

    temp = R.row(1)*terms.xx*R.row(1).transpose();
    M(2,2) = temp;
    temp = -2.0*R.row(0)*terms.xy*R.row(1).transpose();
    M(2,2) += temp;
    temp = R.row(0)*terms.yy*R.row(0).transpose();
    M(2,2) += temp;

    M(1, 0) = M(0, 1);
    M(2, 0) = M(0, 2);
    M(2, 1) = M(1, 2);
}

}
