#ifndef KALMAN_CPP_BASE_UPDATER_H
#define KALMAN_CPP_BASE_UPDATER_H

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

#include "filter_utils.h"

namespace KalmanCpp {

template <typename Derived, typename Scalar, int StateDim, int MeasDim, JacobianMethod Method>
class BaseUpdater {
 // TODO add static assert for type checking
 protected:
  BaseUpdater() noexcept {};

 public:
  typedef Eigen::Matrix<Scalar, StateDim, 1> InputType;
  typedef Eigen::Matrix<Scalar, MeasDim, 1> ValueType;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };

 protected:

  template <typename InMat, typename OutMat>
  void GetMeasurement(const InMat& state, OutMat& measurement) const {
    const Derived* d = static_cast<const Derived*>(this);
    d->template GetPrediction<InMat,OutMat>(state, measurement);
  }

  template <typename InMat, typename OutMat>
  void GetJacobian(const InMat& state, OutMat& jacobian) const {
    const Derived* d = static_cast<const Derived*>(this);
    d->template GetJacobian<InMat,OutMat>(state, jacobian);
  }


 public:

  template <typename InMat, typename OutVec, typename OutMat>
  std::pair<OutVec, OutMat>  Update([[maybe_unused]]const InMat& state) const {
    OutVec measurement;
    OutMat jacobian;
    if constexpr (Method == JacobianMethod::Analytical) {
      GetMeasurement<InMat, OutVec>(state, measurement);
      GetJacobian<InMat, OutMat>(state, jacobian);
    } else {
      Eigen::AutoDiffJacobian<BaseUpdater<Derived,Scalar,StateDim,MeasDim,Method>> auto_differ(*this);
      auto_differ(state, &measurement, &jacobian);
    }
    return {measurement, jacobian};
  }

  // The () operator is overloaded to be used by Eigen::AutoDiffJacobian
  template <typename InMat, typename OutMat>
  void operator()(const InMat& input, OutMat* output) const {
    GetMeasurement<InMat,OutMat>(input, *output);
  };
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_UPDATER_H