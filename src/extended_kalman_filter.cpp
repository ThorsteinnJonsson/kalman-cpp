#include "extended_kalman_filter.h"

#include <Eigen/LU>
#include <unsupported/Eigen/AutoDiff>
// https://stackoverflow.com/questions/39435198/eigens-autodiffjacobian-need-some-help-getting-a-learning-example-to-work

#include <iostream>

using namespace KalmanCpp;

ExtendedKalmanFilter::ExtendedKalmanFilter(size_t state_dim, size_t meas_dim)
    : nx_(state_dim),
      nz_(meas_dim) {
  x_ = Eigen::Vector2f::Zero(nx_);
  P_ = Eigen::MatrixXf::Zero(nx_, nx_);
}

void ExtendedKalmanFilter::Predict(float dt) {
  x_ = f_(x_, dt);
  const auto F = f_jacobian_(x_, dt);
  P_ = F * P_ * F.transpose() + Q_;
}

void ExtendedKalmanFilter::Predict(const Eigen::VectorXf& u, float dt) {
  x_ = f_(x_, dt) + B_ * u;
  const auto F = f_jacobian_(x_, dt);
  P_ = F * P_ * F.transpose() + Q_;
}

void ExtendedKalmanFilter::Update(const Eigen::VectorXf& z) {

  const Eigen::VectorXf y = (residual_.has_value()) ? (*residual_)(z,h_(x_)) : z - h_(x_);
  const auto H = h_jacobian_(x_);
  const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
  const Eigen::MatrixXf I = Eigen::MatrixXf::Identity(nx_, nx_);
  x_ = x_ + K * y;
  P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

void ExtendedKalmanFilter::InitState(const Eigen::VectorXf& state,
                             const Eigen::MatrixXf& cov) {
  x_ = state;
  P_ = cov;
}

void ExtendedKalmanFilter::InitUncertainty(const Eigen::MatrixXf& process_noise,
                                   const Eigen::MatrixXf& measurement_noise) {
  Q_ = process_noise;
  R_ = measurement_noise;
}

void ExtendedKalmanFilter::Print() const {
  std::cout << "Kalman filter:\n";
  std::cout << "x\n" << x_.transpose() << "\n";
  std::cout << "P\n" << P_ << "\n";

  std::cout << "Q\n" << Q_ << "\n";
  std::cout << "R\n" << R_ << "\n";
}