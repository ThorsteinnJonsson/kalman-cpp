#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <functional>
#include <optional>

#include "filter_utils.h"
#include "numerical/jacobian.h"
#include "prediction/base_predictor.h"

namespace KalmanCpp {

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod=JacobianCalculationMethod::Numerical>
class ExtendedKalmanFilter {
 private:
  using StateVec = Eigen::Matrix<T, StateDim, 1>;
  using StateMat = Eigen::Matrix<T, StateDim, StateDim>;
  using MeasVec = Eigen::Matrix<T, MeasDim, 1>;
  using MeasMat = Eigen::Matrix<T, MeasDim, MeasDim>;
  using ObservationModelMat = Eigen::Matrix<T, MeasDim, StateDim>;
  using KalmanGainMat = Eigen::Matrix<T, StateDim, MeasDim>;

 public:
  ExtendedKalmanFilter() = default;

  void Predict(float dt);
  void Predict(const StateVec& u, float dt);
  void Update(const MeasVec& z);
  void NumericalUpdate(const MeasVec& z); // TODO determine if numerical jacobian is used by template 

  void InitState(const StateVec& state, const StateMat& cov);
  void InitUncertainty(const StateMat& process_noise,
                       const MeasMat& measurement_noise);

  // TODO combine SetStateTransition and JacobianSetStateTransition
  void SetStateTransition(std::array<std::function<T(const StateVec&, float dt)>, StateDim>&& funcs) {
    fs_ = funcs;
  }

  void JacobianSetStateTransition(std::array<std::function<Eigen::AutoDiffScalar<Eigen::Matrix<T, StateDim, 1>>(const std::array<Eigen::AutoDiffScalar<Eigen::Matrix<T, StateDim, 1>>, StateDim>&, float)>,StateDim>&& funcs) {
    fsj_ = funcs;
  }

  void SetStateTransitionJacobian(
      std::function<StateMat(const StateVec&, float dt)>&& func) {
    f_jacobian_ = func;
  }

  void SetMeasurementFunction(std::function<MeasVec(const StateVec&)>&& func) {
    h_ = func;
  }
  void SetMeasurementJacobian(std::function<ObservationModelMat(const StateVec&)>&& func) {
    h_jacobian_ = func;
  }

  void SetControlInputFunction(const StateMat& B) {
    B_ = B;
  }

  void SetResidualFunction(
      std::function<MeasVec(const MeasVec&, const MeasVec&)>&& residual_func) {
    residual_ = residual_func;
  }

  StateMat GetStateTransitionJacobian(float dt);

  void SetPredictor(DerivedPredictor<T,StateDim,StateDim>&& predictor) { predictor_ = predictor; }

  const StateVec& State() const { return x_; }
  const StateMat& Uncertainty() const { return P_; }

 private:
  StateVec PredictState(float dt);

 private:
  StateVec x_ = StateVec::Zero();
  StateMat P_ = StateMat::Zero();

  StateMat Q_;
  MeasMat R_;

  std::array<std::function<T(const StateVec&, float dt)>, StateDim> fs_;
  std::array<std::function<Eigen::AutoDiffScalar<Eigen::Matrix<T, StateDim, 1>>(const std::array<Eigen::AutoDiffScalar<Eigen::Matrix<T, StateDim, 1>>, StateDim>&, float)>,StateDim> fsj_;

  std::function<StateMat(const StateVec&, float dt)> f_jacobian_;

  std::function<MeasVec(const StateVec&)> h_; // TODO use numerical
  std::function<ObservationModelMat(const StateVec&)> h_jacobian_;
  std::optional<std::function<MeasVec(const MeasVec&, const MeasVec&)>>
      residual_;

  StateMat B_;  // Control model

  DerivedPredictor<T,StateDim,StateDim> predictor_;
};

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
typename ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::StateVec ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::PredictState(float dt) {
  (void)dt; // TODO use in predictor
  StateVec x_new = predictor_.template Compute<StateVec,StateVec>(x_);
  return x_new;  
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
void ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::Predict(float dt) {
  x_ = PredictState(dt);
  std::cout << "x:\n" << x_ << std::endl;
  const StateMat F = GetStateTransitionJacobian(dt);
  P_ = F * P_ * F.transpose() + Q_;
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
void ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::Predict(const StateVec& u,
                                                         float dt) {
  x_ = PredictState(dt) + B_ * u;
  const StateMat F = GetStateTransitionJacobian(dt);
  P_ = F * P_ * F.transpose() + Q_;
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
void ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::Update(const MeasVec& z) {
  const Eigen::VectorXf y =
      (residual_.has_value()) ? (*residual_)(z, h_(x_)) : z - h_(x_);
  const auto H = h_jacobian_(x_);
  const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
  const Eigen::MatrixXf I = StateMat::Identity();
  x_ = x_ + K * y;
  P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
void ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::NumericalUpdate(const MeasVec& z) {
  const Eigen::VectorXf y =
      (residual_.has_value()) ? (*residual_)(z, h_(x_)) : z - h_(x_);
  const auto H = h_jacobian_(x_);
  const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
  const Eigen::MatrixXf I = StateMat::Identity();
  x_ = x_ + K * y;
  P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
void ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::InitState(
    const StateVec& state, const StateMat& cov) {
  x_ = state;
  P_ = cov;
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
void ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::InitUncertainty(
    const StateMat& process_noise, const MeasMat& measurement_noise) {
  Q_ = process_noise;
  R_ = measurement_noise;
}

template <typename T, int StateDim, int MeasDim, JacobianCalculationMethod JacobianMethod>
typename ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::StateMat ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::GetStateTransitionJacobian(float dt) {
  if constexpr (JacobianMethod == JacobianCalculationMethod::Manual) {
    return f_jacobian_(x_, dt);
  } else {
    (void)dt; // TODO use dt
    Eigen::AutoDiffJacobian<DerivedPredictor<T,StateDim,StateDim>> auto_differ;
    ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::StateVec out;
    ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod>::StateMat jacobian;
    auto_differ(x_, &out, &jacobian);
    std::cout << "out:\n" << out << std::endl;
    return jacobian;
  }
}

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H