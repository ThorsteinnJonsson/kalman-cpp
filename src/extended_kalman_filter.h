#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <functional>
#include <optional>

#include "filter_utils.h"
#include "numerical/jacobian.h"

namespace KalmanCpp {

template <typename T, int StateDim, int MeasDim>
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

  const StateVec& State() const { return x_; }
  const StateMat& Uncertainty() const { return P_; }

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
};

template <typename T, int StateDim, int MeasDim>
void ExtendedKalmanFilter<T, StateDim, MeasDim>::Predict(float dt) {
  StateVec x_new;
  for (int i = 0; i < StateDim; ++i) {
    x_new(i) = fs_[i](x_,dt);
  }
  x_ = x_new;
  const auto F = f_jacobian_(x_, dt);
  const auto F_numerical = Numerical::CalculateJacobian<T,StateDim,StateDim>(x_, fsj_, dt);
  std::cout << "F:\n" << F << std::endl;
  std::cout << "F_numerical:\n" << F_numerical << std::endl;
  std::cout << "\n\n";
  P_ = F * P_ * F.transpose() + Q_;
}

template <typename T, int StateDim, int MeasDim>
void ExtendedKalmanFilter<T, StateDim, MeasDim>::Predict(const StateVec& u,
                                                         float dt) {
  StateVec x_new;
  for (int i = 0; i < StateDim; ++i) {
    x_new(i) = fs_[i](x_,dt);
  }
  x_ = x_new + B_ * u;
  const auto F = f_jacobian_(x_, dt);
  const auto F_numerical = Numerical::CalculateJacobian<T,StateDim,StateDim>(x_, fsj_, dt);
  std::cout << "F:\n" << F << std::endl;
  std::cout << "F_numerical:\n" << F_numerical << std::endl;
  std::cout << "\n\n";

  P_ = F * P_ * F.transpose() + Q_;
}

template <typename T, int StateDim, int MeasDim>
void ExtendedKalmanFilter<T, StateDim, MeasDim>::Update(const MeasVec& z) {
  const Eigen::VectorXf y =
      (residual_.has_value()) ? (*residual_)(z, h_(x_)) : z - h_(x_);
  const auto H = h_jacobian_(x_);
  const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
  const Eigen::MatrixXf I = StateMat::Identity();
  x_ = x_ + K * y;
  P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

template <typename T, int StateDim, int MeasDim>
void ExtendedKalmanFilter<T, StateDim, MeasDim>::NumericalUpdate(const MeasVec& z) {
  const Eigen::VectorXf y =
      (residual_.has_value()) ? (*residual_)(z, h_(x_)) : z - h_(x_);
  const auto H = h_jacobian_(x_);
  const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
  const Eigen::MatrixXf I = StateMat::Identity();
  x_ = x_ + K * y;
  P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

template <typename T, int StateDim, int MeasDim>
void ExtendedKalmanFilter<T, StateDim, MeasDim>::InitState(
    const StateVec& state, const StateMat& cov) {
  x_ = state;
  P_ = cov;
}

template <typename T, int StateDim, int MeasDim>
void ExtendedKalmanFilter<T, StateDim, MeasDim>::InitUncertainty(
    const StateMat& process_noise, const MeasMat& measurement_noise) {
  Q_ = process_noise;
  R_ = measurement_noise;
}

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H