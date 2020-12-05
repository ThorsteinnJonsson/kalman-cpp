#ifndef KALMAN_CPP_BASE_PREDICTOR_H
#define KALMAN_CPP_BASE_PREDICTOR_H

#include <type_traits>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

#include "filter_utils.h"

template <typename T>
struct has_GetPrediction {
  struct dummy {};

  template <typename C, typename P, typename R> static auto test(P* p, R* r) -> decltype(std::declval<C>().GetPrediction(*p, *r), std::true_type());
  template <typename, typename, typename> static std::false_type test(...);

  typedef decltype(test<T, dummy, dummy>(nullptr, nullptr)) type;
  static const bool value =  std::is_same<std::true_type, decltype(test<T,dummy, dummy>(nullptr, nullptr))>::value;
};

template <typename T>
struct has_GetJacobian {
  struct dummy {};

  template <typename C, typename P, typename R> static auto test(P* p, R* r) -> decltype(std::declval<C>().GetJacobian(*p, *r), std::true_type());
  template <typename, typename, typename> static std::false_type test(...);

  typedef decltype(test<T, dummy, dummy>(nullptr, nullptr)) type;
  static const bool value =  std::is_same<std::true_type, decltype(test<T,dummy, dummy>(nullptr, nullptr))>::value;
};

namespace KalmanCpp {

template <typename Derived, typename Scalar, int StateDim, JacobianCalculationMethod Method>
class Predictor {
 private:
  static constexpr void CompileTimeTypeValidation() {
    static_assert(has_GetPrediction<Derived>::value, "Derived predictor does not have a GetPrediction function defined!");
    if constexpr (Method == JacobianCalculationMethod::Analytical) {
      static_assert(has_GetJacobian<Derived>::value, "Derived predictor does not have a GetJacobian function defined!");
    }
  }
 protected:
  Predictor() noexcept {CompileTimeTypeValidation();};
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
    if constexpr (Method == JacobianCalculationMethod::Analytical) {
      OutVec prediction;
      GetPrediction<InMat, OutVec>(in, prediction);
      OutMat jacobian;
      GetJacobian<InMat, OutMat>(in, jacobian);
      return {prediction, jacobian};
    } else {
      Eigen::AutoDiffJacobian<Predictor<Derived,Scalar,StateDim,Method>> auto_differ(*this);
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