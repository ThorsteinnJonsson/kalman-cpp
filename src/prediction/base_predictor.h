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
  OutMat Compute(const InMat& in) const {
    const Derived* d = static_cast<const Derived*>(this);
    // The template keyword below is required to tell the compiler that
    // this is a template. Doing [ OutMat out = d->Compute<InMat,OutMat>(in);  ]
    // results in a compile error. The template keyword tells the compiler that
    // this is a function call.
    OutMat out = d->template Compute<InMat,OutMat>(in);
    return out;
  }

  template <typename InMat, typename OutMat>
  void operator()(const InMat& input, OutMat* output) const {
    *output = Compute<InMat,OutMat>(input);
  };
};

template <typename Scalar, int N_IN, int N_OUT>
struct DerivedPredictor : public BasePredictor<DerivedPredictor<Scalar, N_IN, N_OUT>, Scalar, N_IN, N_OUT> {

  template <typename InMat, typename OutMat>
  OutMat Compute(const InMat& in) const {
    OutMat out;
    float dt = 1.0f; // TODO
    // out << pow(sin(in(0, 0)), 2.) + pow(cos(in(1, 0)), 2.) + 1.0f;
    out(0) = in(0) + in(1) * dt;
    out(1) = in(1);
    return out;
  }
  
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_BASE_PREDICTOR_H