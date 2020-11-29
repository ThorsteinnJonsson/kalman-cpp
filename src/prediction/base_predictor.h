#ifndef KALMAN_CPP_BASE_PREDICTOR_H
#define KALMAN_CPP_BASE_PREDICTOR_H

#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace KalmanCpp {

template <typename Derived, typename Scalar, int N_IN, int N_OUT>
struct BasePredictor {
  typedef Eigen::Matrix<Scalar, N_IN, 1> InputType;
  typedef Eigen::Matrix<Scalar, N_OUT, 1> ValueType;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };

  template <typename InMat, typename OutMat>
  OutMat Predict(const InMat& in) const {
    const Derived* d = static_cast<const Derived*>(this);
    // The template keyword below is required to tell the compiler that
    // this is a template. Doing [ OutMat out = d->Predict<InMat,OutMat>(in);  ]
    // results in a compile error. The template keyword tells the compiler that
    // this is a function call.
    OutMat out = d->template Predict<InMat,OutMat>(in);
    return out;
  }

  template <typename InMat, typename OutMat>
  void operator()(const InMat& input, OutMat* output) const {
    *output = Predict<InMat,OutMat>(input);
  };
};

template <typename Scalar, int N_IN, int N_OUT>
struct DerivedPredictor : public BasePredictor<DerivedPredictor<Scalar, N_IN, N_OUT>, Scalar, N_IN, N_OUT> {

  template <typename InMat, typename OutMat>
  OutMat Predict(const InMat& in) const {
    OutMat out;
    out(0) = in(0) + in(1) * dt_;
    out(1) = in(1);
    return out;
  }

  template <typename InMat, typename OutMat>
  OutMat Jacobian([[maybe_unused]]const InMat& in) const {
    OutMat jacobian = OutMat::Identity();
    jacobian(0, 1) = dt_;
    return jacobian;
  }

  static constexpr float dt_ = 1.0f;
  
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_PREDICTOR_H