#ifndef KALMAN_CPP_BASE_PREDICTOR_H
#define KALMAN_CPP_BASE_PREDICTOR_H


#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

#include "filter_utils.h"
#include "predict/predictor_type_traits.h"


namespace KalmanCpp {

template <typename Derived, typename Scalar, int StateDim, JacobianMethod Method>
class BasePredictor {
 private:
  static constexpr void CompileTimeTypeValidation() {
    static_assert(has_get_prediction<Derived>::value, "Derived predictor does not have a GetPrediction function defined!");
    if constexpr (Method == JacobianMethod::Analytical) {
      static_assert(has_get_jacobian<Derived>::value, "Derived predictor does not have a GetJacobian function defined!");
    }
  }
 protected:
  BasePredictor() noexcept {CompileTimeTypeValidation();};
 public:
  typedef Eigen::Matrix<Scalar, StateDim, 1> InputType;
  typedef Eigen::Matrix<Scalar, StateDim, 1> ValueType;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };
 protected:
  template <typename InMat, typename OutMat>
  void GetPrediction(const InMat& in, OutMat& out) const {
    const Derived* d = static_cast<const Derived*>(this);
    d->template GetPrediction<InMat,OutMat>(in, out);
  }

  template <typename InMat, typename OutMat>
  void GetJacobian(const InMat& in, OutMat& out) const {
    const Derived* d = static_cast<const Derived*>(this);
    d->template GetJacobian<InMat,OutMat>(in, out);
  }

  float Timestep() const {return this->dt_; }

 public:
  template <typename InMat, typename OutVec, typename OutMat>
  std::pair<OutVec, OutMat>  Predict([[maybe_unused]]const InMat& in, float dt) const {
    dt_ = dt;
    if constexpr (Method == JacobianMethod::Analytical) {
      OutVec prediction;
      GetPrediction<InMat, OutVec>(in, prediction);
      OutMat jacobian;
      GetJacobian<InMat, OutMat>(in, jacobian);
      return {prediction, jacobian};
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
    GetPrediction<InMat,OutMat>(input, *output);
  };

 protected:
  mutable float dt_ = 0.0f;
};


}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_PREDICTOR_H