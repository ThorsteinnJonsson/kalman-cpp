#ifndef KALMAN_CPP_BASE_PREDICTOR_H
#define KALMAN_CPP_BASE_PREDICTOR_H

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

#include "filter_utils.h"

namespace KalmanCpp {

template <typename Derived, typename Scalar, int StateDim, JacobianCalculationMethod Method>
class BasePredictor {
 public:
  typedef Eigen::Matrix<Scalar, StateDim, 1> InputType;
  typedef Eigen::Matrix<Scalar, StateDim, 1> ValueType;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };
 protected:
  template <typename InMat, typename OutMat>
  OutMat GetPrediction(const InMat& in) const {
    const Derived* d = static_cast<const Derived*>(this);
    // The template keyword below is required to tell the compiler that
    // this is a template. Doing [ OutMat out = d->GetPrediction<InMat,OutMat>(in);  ]
    // results in a compile error. The template keyword tells the compiler that
    // this is a function call.
    OutMat out = d->template GetPrediction<InMat,OutMat>(in);
    return out;
  }

  template <typename InMat, typename OutMat>
  OutMat GetJacobian(const InMat& in) const {
    // TODO this gives a runtime error when the GetJacobian method is not defined in the 
    // derived class. Need to change it to compile-time error
    const Derived* d = static_cast<const Derived*>(this);
    OutMat out = d->template GetJacobian<InMat,OutMat>(in);
    return out;
  }

 public:
  template <typename InMat, typename OutVec, typename OutMat>
  std::pair<OutVec, OutMat>  Predict([[maybe_unused]]const InMat& in, float dt) const {
    dt_ = dt;
    if constexpr (Method == JacobianCalculationMethod::Analytical) {
      return {GetPrediction<InMat, OutVec>(in), GetJacobian<InMat, OutMat>(in)};
    } else {
      Eigen::AutoDiffJacobian<BasePredictor<Derived,Scalar,StateDim,Method>> auto_differ(*this);
      OutVec prediction;
      OutMat jacobian;
      auto_differ(in, &prediction, &jacobian);
      return {prediction, jacobian};
    }
  }

  template <typename InMat, typename OutMat>
  void operator()(const InMat& input, OutMat* output) const {
    *output = GetPrediction<InMat,OutMat>(input);
  };

 protected:
  mutable float dt_ = 0.0f;
};

template <typename Scalar, int StateDim, JacobianCalculationMethod Method>
class Predictor : public BasePredictor<Predictor<Scalar, StateDim,Method>, Scalar, StateDim, Method> {
  friend class BasePredictor<Predictor<Scalar, StateDim,Method>, Scalar, StateDim, Method>;
 protected:
  float Timestep() const { return BasePredictor<Predictor<Scalar, StateDim,Method>, Scalar, StateDim, Method>::dt_; }

  template <typename InMat, typename OutMat>
  OutMat GetPrediction(const InMat& in) const {
    OutMat out;
    out(0) = in(0) + in(1) * Timestep();
    out(1) = in(1);
    return out;
  }

  template <typename InMat, typename OutMat>
  OutMat GetJacobian([[maybe_unused]]const InMat& in) const {
    OutMat jacobian = OutMat::Identity();
    jacobian(0, 1) = Timestep();
    return jacobian;
  }

};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_PREDICTOR_H