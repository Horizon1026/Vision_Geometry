#include "geometry_epipolar.h"

namespace vision_geometry {

void EpipolarSolver::DecomposeEssentialMatrix(const Mat3 &essential, Mat3 &R0, Mat3 &R1, Vec3 &t0, Vec3 &t1) {
    Eigen::JacobiSVD<Mat3> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 Vt = svd.matrixV().transpose();

    // Make sure the determinants of U and V are both positive.
    if (U.determinant() < 0) {
        U = -U;
    }
    if (Vt.determinant() < 0) {
        Vt = -Vt;
    }

    // R means R(PI / 2).
    Mat3 R = Mat3::Zero();
    R << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R0 = U * R * Vt;
    R1 = U * R.transpose() * Vt;
    t0 = U.col(2);
    t1 = -t0;
}

void EpipolarSolver::RefineEssentialMatrix(Mat3 &essential) {
    Eigen::JacobiSVD<Mat3> svd(essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();

    // Enforce rotation matrices (det = +1).
    if (U.determinant() < 0) {
        U = -U;
    }
    if (V.determinant() < 0) {
        V = -V;
    }

    essential = U * Vec3(1, 1, 0).asDiagonal() * V.transpose();
}

}  // namespace vision_geometry
