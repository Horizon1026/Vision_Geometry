#ifndef _GEOMETRY_PNP_H_
#define _GEOMETRY_PNP_H_

#include "datatype_basic.h"

namespace VISION_GEOMETRY {

class PnpSolver {

public:
enum PnpMethod : uint8_t {
    PNP_ALL = 0,
    PNP_RANSAC,
    PNP_HUBER,
    PNP_CAUCHY,
};

enum PnpResult : uint8_t {
    SOlVED = 0,
    UNSOLVED,
    LARGE_RISIDUAL,
};

typedef struct {
    uint32_t kMaxSolvePointsNumber = 200;
    uint32_t kMaxIteration = 10;
    float kMaxConvergeStep = 1e-6f;
    float kMaxConvergeResidual = 1e-3f;
    float kMinRansacInlierRatio = 0.9f;
    float kMinValidDepth = 1e-3f;
    PnpMethod kMethod = PNP_RANSAC;
} PnpOptions;

public:
    explicit PnpSolver() = default;
    virtual ~PnpSolver() = default;

    bool EstimatePose(const std::vector<Vec3> &p_w,
                      const std::vector<Vec2> &norm_uv,
                      Quat &q_wc,
                      Vec3 &p_wc,
                      std::vector<PnpResult> &status);

    PnpOptions &options() { return options_;}

private:
    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w,
                            const std::vector<Vec2> &norm_uv,
                            Quat &q_wc,
                            Vec3 &p_wc,
                            std::vector<PnpResult> &status);

    bool EstimatePoseUseAll(const std::vector<Vec3> &p_w,
                            const std::vector<Vec2> &norm_uv,
                            Quat &q_wc,
                            Vec3 &p_wc);

    bool EstimatePoseRANSAC(const std::vector<Vec3> &p_w,
                            const std::vector<Vec2> &norm_uv,
                            Quat &q_wc,
                            Vec3 &p_wc,
                            std::vector<PnpResult> &status);

    inline float Huber(float param, float x) {
        float huber = 1.0f;
        if (x > param) {
            huber = 2.0f * std::sqrt(x) * param - param * param;
            huber /= x;
        }
        return huber;
    }

    inline float Cauchy(float param, float x) {
        float cauchy = 1.0f;
        float param2 = param * param;
        if (x > param) {
            cauchy = param2 * std::log(1.0f / param2 + 1.0f);
            cauchy /= x;
        }
        return cauchy;
    }

private:
    PnpOptions options_;
};
}

#endif // end of _GEOMETRY_PNP_H_
