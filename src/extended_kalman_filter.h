#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <functional>
#include <optional>
#include <memory>

#include "filter_utils.h"
#include "predict/base_predictor.h"
#include "update/base_updater.h"

namespace KalmanCpp {

template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater> // TODO check if T is floating
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

  void InitState(const StateVec& state, const StateMat& cov);
  void InitUncertainty(const StateMat& process_noise,
                       const MeasMat& measurement_noise);


  void SetControlInputFunction(const StateMat& B) {
    B_ = B;
  }

  void SetPredictor(std::unique_ptr<TPredictor>&& predictor) { predictor_ = std::move(predictor); }
  void SetUpdater(std::unique_ptr<TUpdater>&& updater) {updater_ = std::move(updater); }
  

  const StateVec& State() const { return x_; }
  const StateMat& Uncertainty() const { return P_; }

 private:
  StateVec x_ = StateVec::Zero();
  StateMat P_ = StateMat::Zero();

  StateMat Q_;
  MeasMat R_;

  StateMat B_;  // Control model

  std::unique_ptr<TPredictor> predictor_;

  std::unique_ptr<TUpdater> updater_;
};

template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
void ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater>::Predict(float dt) {
  auto [x_new, jacobian] = predictor_->template Predict<StateVec,StateVec,StateMat>(x_, dt);
  x_ = x_new;
  P_ = jacobian * P_ * jacobian.transpose() + Q_;
}

template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
void ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater>::Predict(const StateVec& u, float dt) {
  Predict(dt);
  x_ += B_ * u;
}

template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
void ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater>::Update(const MeasVec& z) {
  auto [z_hat, H] = updater_->template Update<StateVec,MeasVec,ObservationModelMat>(x_);
  const Eigen::VectorXf y = z - z_hat;
  const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
  const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
  const Eigen::MatrixXf I = StateMat::Identity();
  x_ = x_ + K * y;
  P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
}

template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
void ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater>::InitState(
    const StateVec& state, const StateMat& cov) {
  x_ = state;
  P_ = cov;
}

template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
void ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater>::InitUncertainty(
    const StateMat& process_noise, const MeasMat& measurement_noise) {
  Q_ = process_noise;
  R_ = measurement_noise;
}

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H