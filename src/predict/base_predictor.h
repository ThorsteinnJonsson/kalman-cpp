#ifndef KALMAN_CPP_BASE_PREDICTOR_H
#define KALMAN_CPP_BASE_PREDICTOR_H


#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

#include "filter_utils.h"
#include "internal/type_traits.h"


namespace KalmanCpp {

template <typename Derived, typename Scalar, int StateDim, JacobianMethod Method>
class BasePredictor {
 private:
  static constexpr void CompileTimeTypeValidation() {
    static_assert(type_traits::has_get_prediction_v<Derived>, "Derived predictor does not have a GetPrediction function defined!");
    if constexpr (Method == JacobianMethod::Analytical) {
      static_assert(type_traits::has_get_jacobian_v<Derived>, "Derived predictor does not have a GetJacobian function defined!");
    }
  }
 protected:
  BasePredictor() noexcept {CompileTimeTypeValidation();}

 public:
  typedef Eigen::Matrix<Scalar, StateDim, 1> InputType;
  typedef Eigen::Matrix<Scalar, StateDim, 1> ValueType;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };
  
 protected:

  template <typename InMat, typename OutMat>
  void GetPrediction(const InMat& state, OutMat& prediction) const {
    const Derived* d = static_cast<const Derived*>(this);
    d->template GetPrediction<InMat,OutMat>(state, prediction);
  }

  template <typename InMat, typename OutMat>
  void GetJacobian(const InMat& state, OutMat& jacobian) const {
    const Derived* d = static_cast<const Derived*>(this);
    d->template GetJacobian<InMat,OutMat>(state, jacobian);
  }

  float Timestep() const {return this->dt_; }

 public:

  template <typename InMat, typename OutVec, typename OutMat>
  std::pair<OutVec, OutMat>  Predict([[maybe_unused]]const InMat& state, float dt) const {
    dt_ = dt;
    OutVec prediction;
    OutMat jacobian;
    if constexpr (Method == JacobianMethod::Analytical) {
      GetPrediction<InMat, OutVec>(state, prediction);
      GetJacobian<InMat, OutMat>(state, jacobian);
    } else {
      Eigen::AutoDiffJacobian<BasePredictor<Derived,Scalar,StateDim,Method>> auto_differ(*this);
      auto_differ(state, &prediction, &jacobian);
    }
    return {prediction, jacobian};
  }

  // The () operator is overloaded to be used by Eigen::AutoDiffJacobian
  template <typename InMat, typename OutMat>
  void operator()(const InMat& state, OutMat* output) const {
    GetPrediction<InMat,OutMat>(state, *output);
  };

 protected:
  mutable float dt_ = 0.0f;
};


}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_PREDICTOR_H