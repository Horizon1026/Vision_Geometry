#ifndef _GEOMETRY_EPIPOLAR_H_
#define _GEOMETRY_EPIPOLAR_H_

#include "basic_type.h"

namespace vision_geometry {

/* Class Epipolar Solver Declaration. */
class EpipolarSolver {

public:
    enum class EpipolarMethod : uint8_t {
        kUseAll = 0,
        kRansac = 1,
    };

    enum class EpipolarModel : uint8_t {
        kFivePoints = 0,
        kEightPoints = 1,
    };

    enum class EpipolarResult : uint8_t {
        kUnsolved = 0,
        kSolved = 1,
        kLargeResidual = 2,
    };

    struct EpipolarOptions {
        uint32_t kMaxSolvePointsNumber = 200;
        uint32_t kMaxIteration = 200;
        float kMaxEpipolarResidual = 1e-3f;
        float kMinRansacInlierRatio = 0.9f;
        EpipolarMethod kMethod = EpipolarMethod::kRansac;
        EpipolarModel kModel = EpipolarModel::kFivePoints;
    };

public:
    EpipolarSolver() = default;
    virtual ~EpipolarSolver() = default;

    // For RANSAC, a nonempty status selects kSolved (tracked) pairs; other entries are preserved.
    // Bearing overloads normalize camera-frame rays without projecting onto the image plane.
    bool EstimateEssential(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);
    bool EstimateEssential(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential, std::vector<uint8_t> &status);

    bool RecoverPoseFromEssential(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential, Mat3 &R_cr, Vec3 &t_cr);
    bool RecoverPoseFromEssential(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const Mat3 &essential, Mat3 &R_cr, Vec3 &t_cr);

    // Reference for member variables.
    EpipolarOptions &options() { return options_; }

    // Const reference for member variables.
    const EpipolarOptions &options() const { return options_; }

private:
    bool EstimateEssentialUseAll(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);
    bool EstimateEssentialUseAll(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential, std::vector<uint8_t> &status);

    bool EstimateEssentialUseAll(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential);
    bool EstimateEssentialUseAll(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential);

    bool EstimateEssentialRansac(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);
    bool EstimateEssentialRansac(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential, std::vector<uint8_t> &status);

    void ComputeEssentialModelResidual(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential,
                                       std::vector<float> &residuals);
    void ComputeEssentialModelResidual(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const Mat3 &essential,
                                       std::vector<float> &residuals);

    float ComputeEssentialModelResidualSummary(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential);
    float ComputeEssentialModelResidualSummary(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const Mat3 &essential);

    void CheckEssentialPairsStatus(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);
    void CheckEssentialPairsStatus(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential, std::vector<uint8_t> &status);

    void DecomposeEssentialMatrix(const Mat3 &essential, Mat3 &R0, Mat3 &R1, Vec3 &t0, Vec3 &t1);

private:
    /* Method for five points model. */
    bool EstimateEssentialUseFivePoints(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential);
    bool EstimateEssentialUseFivePoints(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential);

    bool SolveFivePointConstraint(const Eigen::Matrix<float, Eigen::Dynamic, 9> &Q, std::vector<Mat3> &essentials);

    void GaussJordanElimination(float *e, float *A);

    bool FindBestOneFromAllPossibleEssentials(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const std::vector<Mat3> &essentials,
                                              Mat3 &best_essential);
    bool FindBestOneFromAllPossibleEssentials(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, const std::vector<Mat3> &essentials,
                                              Mat3 &best_essential);

private:
    /* Method for eight points model. */
    bool EstimateEssentialUseEightPoints(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential);
    bool EstimateEssentialUseEightPoints(const std::vector<Vec3> &ref_bearing_xyz, const std::vector<Vec3> &cur_bearing_xyz, Mat3 &essential);

    bool SolveEightPointConstraint(const Mat &A, Mat3 &essential, bool check_rank);

    void RefineEssentialMatrix(Mat3 &essential);

private:
    EpipolarOptions options_;
};

}  // namespace vision_geometry

#endif  // _GEOMETRY_EPIPOLAR_H_
