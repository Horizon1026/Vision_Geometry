#include "geometry_pnp.h"

namespace vision_geometry {

float PnpSolver::Huber(float param, float x) {
    float huber = 1.0f;
    if (x > param) {
        huber = param / x;
    }
    return huber;
}

float PnpSolver::Cauchy(float param, float x) {
    float cauchy = 1.0f;
    float param2 = param * param;
    if (x > param) {
        cauchy = param2 / (param2 + x * x);
    }
    return cauchy;
}

}  // namespace vision_geometry
