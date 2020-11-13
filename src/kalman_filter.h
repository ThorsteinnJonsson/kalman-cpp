#ifndef KFCPP_KALMAN_FILTER
#define KFCPP_KALMAN_FILTER

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

namespace KalmanCpp {

template <typename T, int StateDim, int MeasDim>  // TODO check if T is floating
                                                  // point type
                                                  class KalmanFilter {
 private:
  using StateVec = Eigen::Matrix<T, StateDim, 1>;
  using StateMat = Eigen::Matrix<T, StateDim, StateDim>;
  using MeasVec = Eigen::Matrix<T, MeasDim, 1>;
  using MeasMat = Eigen::Matrix<T, MeasDim, MeasDim>;
  using ObservationModelMat = Eigen::Matrix<T, MeasDim, StateDim>;
  using KalmanGainMat = Eigen::Matrix<T, StateDim, MeasDim>;

 public:
  KalmanFilter() = default;

  void Predict();
  void Update(const MeasVec& z);

  void InitState(const StateVec& state, const StateMat& cov);
  void InitUncertainty(const StateMat& process_noise,
                       const MeasMat& measurement_noise);
  void SetStateTransitionMatrix(const StateMat& state_transition_matrix);
  void SetMeasurementFunctionMatrix(
      const ObservationModelMat& meas_function_matrix);

  const StateVec& State() const { return x_; }
  const StateMat& Uncertainty() const { return P_; }

  void Print() const;

 private:
  StateVec x_ = StateVec::Zero();
  StateMat P_ = StateMat::Zero();

  StateMat Q_;
  MeasMat R_;

  StateMat F_;
  ObservationModelMat H_;
};

template <typename T, int StateDim, int MeasDim>
void KalmanFilter<T, StateDim, MeasDim>::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

template <typename T, int StateDim, int MeasDim>
void KalmanFilter<T, StateDim, MeasDim>::Update(const MeasVec& z) {
  const MeasVec y = z - H_ * x_;
  const MeasMat S = (H_ * P_ * H_.transpose() + R_);
  const KalmanGainMat K = P_ * H_.transpose() * S.inverse();
  const StateMat I = StateMat::Identity();

  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * R_ * K.transpose();
}

template <typename T, int StateDim, int MeasDim>
void KalmanFilter<T, StateDim, MeasDim>::InitState(const StateVec& state,
                                                   const StateMat& cov) {
  x_ = state;
  P_ = cov;
}

template <typename T, int StateDim, int MeasDim>
void KalmanFilter<T, StateDim, MeasDim>::InitUncertainty(
    const StateMat& process_noise, const MeasMat& measurement_noise) {
  Q_ = process_noise;
  R_ = measurement_noise;
}

template <typename T, int StateDim, int MeasDim>
void KalmanFilter<T, StateDim, MeasDim>::SetStateTransitionMatrix(
    const StateMat& state_transition_matrix) {
  F_ = state_transition_matrix;
}

template <typename T, int StateDim, int MeasDim>
void KalmanFilter<T, StateDim, MeasDim>::SetMeasurementFunctionMatrix(
    const ObservationModelMat& meas_function_matrix) {
  H_ = meas_function_matrix;
}

}  // namespace KalmanCpp

#endif  // KFCPP_KALMAN_FILTER