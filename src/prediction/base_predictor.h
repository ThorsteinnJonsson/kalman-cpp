#ifndef KALMAN_CPP_BASE_PREDICTOR_H
#define KALMAN_CPP_BASE_PREDICTOR_H

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

#include "filter_utils.h"

namespace KalmanCpp {

template <typename Derived, typename Scalar, int StateDim>
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
  
 public:
  template <typename InMat, typename OutMat>
  void operator()(const InMat& input, OutMat* output) const {
    *output = GetPrediction<InMat,OutMat>(input);
  };
};

template <typename Scalar, int StateDim, JacobianCalculationMethod Method=JacobianCalculationMethod::Numerical>
class DerivedPredictor : public BasePredictor<DerivedPredictor<Scalar, StateDim>, Scalar, StateDim> {
  friend class BasePredictor<DerivedPredictor<Scalar, StateDim>, Scalar, StateDim>;
 protected:
  template <typename InMat, typename OutMat>
  OutMat GetPrediction(const InMat& in) const {
    OutMat out;
    out(0) = in(0) + in(1) * dt_;
    out(1) = in(1);
    return out;
  }

  template <typename InMat, typename OutMat>
  OutMat GetJacobian([[maybe_unused]]const InMat& in) const {
    OutMat jacobian = OutMat::Identity();
    jacobian(0, 1) = dt_;
    return jacobian;
  }

 public:
  template <typename InMat, typename OutVec, typename OutMat>
  std::pair<OutVec, OutMat>  Predict([[maybe_unused]]const InMat& in) const {
    if constexpr (Method == JacobianCalculationMethod::Analytical) {
      return {GetPrediction<InMat, OutVec>(in), GetJacobian<InMat, OutMat>(in)};
    } else {
      Eigen::AutoDiffJacobian<DerivedPredictor<Scalar,StateDim,Method>> auto_differ(*this);
      OutVec prediction;
      OutMat jacobian;
      auto_differ(in, &prediction, &jacobian);
      return {prediction, jacobian};
    }
  }

  static constexpr float dt_ = 1.0f;
  
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_PREDICTOR_H