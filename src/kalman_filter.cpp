#include "kalman_filter.h"

#include <Eigen/LU>
#include <iostream>

using namespace KalmanCpp;

KalmanFilter::KalmanFilter(size_t state_dim, size_t meas_dim)
    : nx_(state_dim),
      nz_(meas_dim) {
  x_ = Eigen::Vector2f::Zero(nx_);
  P_ = Eigen::MatrixXf::Zero(nx_, nx_);
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const Eigen::VectorXf& z) {
  const Eigen::VectorXf y = z - H_ * x_;
  const Eigen::MatrixXf S = (H_ * P_ * H_.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse();
  const Eigen::MatrixXf I = Eigen::MatrixXf::Identity(nx_, nx_);

  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * R_ * K.transpose();
}

void KalmanFilter::InitState(const Eigen::VectorXf& state,
                             const Eigen::MatrixXf& cov) {
  x_ = state;
  P_ = cov;
}

void KalmanFilter::InitUncertainty(const Eigen::MatrixXf& process_noise,
                                   const Eigen::MatrixXf& measurement_noise) {
  Q_ = process_noise;
  R_ = measurement_noise;
}

void KalmanFilter::SetStateTransitionMatrix(
    const Eigen::MatrixXf& state_transition_matrix) {
  F_ = state_transition_matrix;
}

void KalmanFilter::SetMeasurementFunctionMatrix(
    const Eigen::MatrixXf& meas_function_matrix) {
  H_ = meas_function_matrix;
}

void KalmanFilter::Print() const {
  std::cout << "Kalman filter:\n";
  std::cout << "x\n" << x_.transpose() << "\n";
  std::cout << "P\n" << P_ << "\n";

  std::cout << "Q\n" << Q_ << "\n";
  std::cout << "R\n" << R_ << "\n";

  std::cout << "F\n" << F_ << "\n";
  std::cout << "H\n" << H_ << "\n";
}