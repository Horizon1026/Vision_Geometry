#ifndef _EIGEN_OPTIMIZATION_FUNCTOR_H_
#define _EIGEN_OPTIMIZATION_FUNCTOR_H_

#include "basic_type.h"
#include "eigen3/Eigen/src/Core/util/DisableStupidWarnings.h"

namespace VISION_GEOMETRY {

/**
 * Generic functor base for use with the Eigen-nonlinear optimization
 * toolbox. Please refer to the Eigen-documentation for further information.
 */
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct OptimizationFunctor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };

    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    const int m_inputs;
    const int m_values;

    OptimizationFunctor() :
        m_inputs(InputsAtCompileTime),
        m_values(ValuesAtCompileTime) {}
    OptimizationFunctor(int inputs, int values) :
        m_inputs(inputs), m_values(values) {}

    int inputs() const {
        return m_inputs;
    }

    int values() const {
        return m_values;
    }
};

}

#endif // end of _EIGEN_OPTIMIZATION_FUNCTOR_H_
