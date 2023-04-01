#ifndef _GEOMETRY_EPIPOLAR_H_
#define _GEOMETRY_EPIPOLAR_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

class EpipolarSolver {

public:
    enum class EpipolarMethod : uint8_t {
        EPIPOLAR_ALL = 0,
        EPIPOLAR_RANSAC = 1,
        EPIPOLAR_HUBER = 2,
        EPIPOLAR_CAUCHY = 3,
    };

    enum class EpipolarModel : uint8_t {
        FIVE_POINTS = 0,
        EIGHT_POINTS = 1,
    };

    enum class EpipolarResult : uint8_t {
        UNSOLVED = 0,
        SOLVED = 1,
        LARGE_RISIDUAL = 2,
    };

    struct EpipolarOptions {
        uint32_t kMaxSolvePointsNumber = 200;
        uint32_t kMaxIteration = 10;
        float kMaxEpipolarResidual = 1e-3f;
        float kMinRansacInlierRatio = 0.9f;
        EpipolarMethod kMethod = EpipolarMethod::EPIPOLAR_ALL;
        EpipolarModel kModel = EpipolarModel::FIVE_POINTS;
    };

public:
    explicit EpipolarSolver() = default;
    virtual ~EpipolarSolver() = default;

    bool EstimateEssential(const std::vector<Vec2> &norm_uv_ref,
                           const std::vector<Vec2> &norm_uv_cur,
                           Mat3 &essential,
                           std::vector<EpipolarResult> &status);

    void DecomposeEssentialMatrix(const Mat3 &essential, Mat3 &R0, Mat3 &R1, Vec3 &t0, Vec3 &t1);

    EpipolarOptions &options() { return options_; }

private:
    bool EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                 const std::vector<Vec2> &norm_uv_cur,
                                 Mat3 &essential,
                                 std::vector<EpipolarResult> &status);

    bool EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                 const std::vector<Vec2> &norm_uv_cur,
                                 Mat3 &essential);

    bool EstimateEssentialRansac(const std::vector<Vec2> &norm_uv_ref,
                                 const std::vector<Vec2> &norm_uv_cur,
                                 Mat3 &essential,
                                 std::vector<EpipolarResult> &status);

    void ComputeEssentialModelResidual(const std::vector<Vec2> &norm_uv_ref,
                                       const std::vector<Vec2> &norm_uv_cur,
                                       const Mat3 &essential,
                                       std::vector<float> &residuals);

private:
    /* Method for five points model. */
    bool EstimateEssentialUseFivePoints(const std::vector<Vec2> &norm_uv_ref,
                                        const std::vector<Vec2> &norm_uv_cur,
                                        Mat3 &essential);

    void GaussJordanElimination(float *e, float *A);

    void FindBestOneFromAllPossibleEssentials(const std::vector<Vec2> &norm_uv_ref,
                                              const std::vector<Vec2> &norm_uv_cur,
                                              const std::vector<Mat3> &essentials,
                                              Mat3 &best_essential);

private:
    /* Method for eight points model. */
    bool EstimateEssentialUseEightPoints(const std::vector<Vec2> &norm_uv_ref,
                                         const std::vector<Vec2> &norm_uv_cur,
                                         Mat3 &essential);

    void RefineEssentialMatrix(Mat3 &essential);

private:
    EpipolarOptions options_;
    Mat A;
};

}

#endif // _GEOMETRY_EPIPOLAR_H_
