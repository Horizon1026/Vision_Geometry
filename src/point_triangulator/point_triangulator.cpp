#include "point_triangulator.h"

namespace vision_geometry {

float PointTriangulator::Huber(float param, float x) {
    float huber = 1.0f;
    if (x > param) {
        huber = param / x;
    }
    return huber;
}

float PointTriangulator::Cauchy(float param, float x) {
    const float param2 = param * param;
    return param2 / (param2 + x * x);
}

}  // namespace vision_geometry
