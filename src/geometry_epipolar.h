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

    struct EpipolarOptions {
        EpipolarMethod kMethod = EpipolarMethod::EPIPOLAR_RANSAC;
    };

public:
    explicit EpipolarSolver() = default;
    virtual ~EpipolarSolver() = default;

    bool EstimateEssential(const std::vector<Vec2> &norm_uv_ref,
                           const std::vector<Vec2> &norm_uv_cur,
                           Mat3 essential);

    EpipolarOptions &options() { return options_; }

private:
    bool EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                 const std::vector<Vec2> &norm_uv_cur,
                                 Mat3 essential,
                                 std::vector<PnpResult> &status);

    bool EstimateEssentialUseAll(const std::vector<Vec2> &norm_uv_ref,
                                 const std::vector<Vec2> &norm_uv_cur,
                                 Mat3 essential);

    bool EstimateEssentialRansac(const std::vector<Vec2> &norm_uv_ref,
                                 const std::vector<Vec2> &norm_uv_cur,
                                 Mat3 essential,
                                 std::vector<PnpResult> &status);

private:
    EpipolarOptions options_;
};

}

#endif // _GEOMETRY_EPIPOLAR_H_
