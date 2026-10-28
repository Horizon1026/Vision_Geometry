#include "relative_rotation.h"

#include "slam_basic_math.h"
#include "slam_operations.h"

namespace vision_geometry {

Vec3 RelativeRotation::EstimateRotationUseAll(const SummationTerms &terms, Quat &q_cr) {
    // Prepare for optimizaiton.
    const int n = 3;
    Vec x = Vec::Zero(n);
    x = Utility::ConvertRotationMatrixToCayley(q_cr.matrix());

    // Use LM solver in eigen3 to optimize x.
    EigenSolverStep functor(terms);
    Eigen::NumericalDiff<EigenSolverStep> num_diff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<EigenSolverStep>, float> solver(num_diff);
    solver.resetParameters();
    solver.parameters.ftol = 5e-5f;
    solver.parameters.xtol = Eigen::NumTraits<float>::epsilon();
    solver.parameters.maxfev = 100;
    solver.minimize(x);

    // Recovery rotation matrix from cayley.
    Vec3 cayley = x;
    const Mat3 R_cr = Utility::ConvertCayleyToRotationMatrix(cayley);
    q_cr = Quat(R_cr);

    return cayley;
}

void RelativeRotation::EstimatePoseUseAll(const SummationTerms &terms, Vec3 &t_cr, Quat &q_cr) {
    // Estiamte rotation with cayley format.
    Vec3 cayley = EstimateRotationUseAll(terms, q_cr);

    // Compute matrix M to recovery translation.
    Mat3 M = Mat3::Zero();
    ComputeM(terms, cayley, M);

    // Decompose matrix M.
    Eigen::EigenSolver<Mat3> eig(M, true);
    Eigen::Matrix<std::complex<float>, 3, 1> D_complex = eig.eigenvalues();
    Eigen::Matrix<std::complex<float>, 3, 3> V_complex = eig.eigenvectors();

    // Sort eigen vectors with values.
    std::map<float, Vec3> eigen_values_vectors;
    for (uint32_t i = 0; i < 3; ++i) {
        Vec3 eigen_vector = Vec3::Zero();
        for (uint32_t j = 0; j < 3; ++j) {
            eigen_vector(j) = V_complex(j, i).real();
        }
        eigen_values_vectors.insert(std::make_pair(D_complex[i].real(), eigen_vector));
    }

    // Record sorted eigen vectors and values.
    Vec3 eigen_values = Vec3::Zero();
    Mat3 eigen_vectors = Mat3::Zero();
    uint32_t idx = 0;
    for (const auto &eigen_value_vector: eigen_values_vectors) {
        eigen_values(idx) = eigen_value_vector.first;
        eigen_vectors.col(idx) = eigen_value_vector.second;
        ++idx;
    }

    // Compute translation, but it seems not good.
    t_cr = eigen_values.head<2>().norm() * eigen_vectors.col(2);
}

}  // namespace vision_geometry
