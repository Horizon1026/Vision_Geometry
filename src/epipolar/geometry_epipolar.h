#ifndef _GEOMETRY_EPIPOLAR_H_
#define _GEOMETRY_EPIPOLAR_H_

#include "basic_type.h"

namespace VISION_GEOMETRY {

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
        uint32_t kMaxIteration = 10;
        float kMaxEpipolarResidual = 1e-3f;
        float kMinRansacInlierRatio = 0.9f;
        EpipolarMethod kMethod = EpipolarMethod::kRansac;
        EpipolarModel kModel = EpipolarModel::kFivePoints;
    };

public:
    EpipolarSolver() = default;
    virtual ~EpipolarSolver() = default;

    bool EstimateEssential(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);

    bool RecoverPoseFromEssential(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential, Mat3 &R_cr, Vec3 &t_cr);

    // Reference for member variables.
    EpipolarOptions &options() { return options_; }

    // Const reference for member variables.
    const EpipolarOptions &options() const { return options_; }

private:
    bool EstimateEssentialUseAll(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);

    bool EstimateEssentialUseAll(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential);

    bool EstimateEssentialRansac(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);

    void ComputeEssentialModelResidual(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential,
                                       std::vector<float> &residuals);

    float ComputeEssentialModelResidualSummary(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const Mat3 &essential);

    void CheckEssentialPairsStatus(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential, std::vector<uint8_t> &status);

    void DecomposeEssentialMatrix(const Mat3 &essential, Mat3 &R0, Mat3 &R1, Vec3 &t0, Vec3 &t1);

private:
    /* Method for five points model. */
    bool EstimateEssentialUseFivePoints(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential);

    void GaussJordanElimination(float *e, float *A);

    void FindBestOneFromAllPossibleEssentials(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, const std::vector<Mat3> &essentials,
                                              Mat3 &best_essential);

private:
    /* Method for eight points model. */
    bool EstimateEssentialUseEightPoints(const std::vector<Vec2> &ref_norm_xy, const std::vector<Vec2> &cur_norm_xy, Mat3 &essential);

    void RefineEssentialMatrix(Mat3 &essential);

private:
    EpipolarOptions options_;
    Mat A;
};

}  // namespace VISION_GEOMETRY

#endif  // _GEOMETRY_EPIPOLAR_H_
