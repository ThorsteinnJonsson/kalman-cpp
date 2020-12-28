#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <memory>

#include "components/base_predictor.h"
#include "components/base_updater.h"

namespace KalmanCpp {

template <typename T, 
          int StateDim, 
          int MeasDim, 
          typename TPredictor, 
          typename TUpdater,
          typename = std::enable_if_t<std::is_floating_point<T>::value>>
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

  void Predict(float dt) {
    auto [x_new, jacobian] = predictor_->template Predict<StateVec,StateVec,StateMat>(x_, dt);
    x_ = x_new;
    P_ = jacobian * P_ * jacobian.transpose() + Q_;    
  }


  void Predict(const StateVec& u, float dt) {
    static_assert(type_traits::has_apply_control_input_v<type_traits::smart_pointer_t<decltype(predictor_)>>, 
                                      "You need to define the GetControlInput member function of your predictor "
                                      "to be able to use the Predict function with control input.");
    Predict(dt);
    StateVec control; 
    predictor_->template GetControlInput<StateVec>(u, control);
    x_ += control;
  }


  void Update(const MeasVec& z) {
    auto [z_hat, H] = updater_->template Update<StateVec,MeasVec,ObservationModelMat>(x_);
    const Eigen::VectorXf y = z - z_hat;
    const Eigen::MatrixXf S = (H * P_ * H.transpose() + R_);
    const Eigen::MatrixXf K = P_ * H.transpose() * S.inverse();
    const Eigen::MatrixXf I = StateMat::Identity();
    x_ = x_ + K * y;
    P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R_ * K.transpose();
  }


  void InitState(const StateVec& state, const StateMat& cov) {
    x_ = state;
    P_ = cov;
  }


  void InitUncertainty(const StateMat& process_noise,
                       const MeasMat& measurement_noise) {
    Q_ = process_noise;
    R_ = measurement_noise;
  }


  void SetPredictor(std::unique_ptr<TPredictor>&& predictor) { 
    static_assert(TPredictor::InputsAtCompileTime == StateDim, 
            "Predictor dimensions do not match the Kalman filter dimensions");
    static_assert(std::is_same<typename TPredictor::ScalarType, T>::value, 
            "Predictor scalar type does not match the Kalman filter type.");
    predictor_ = std::move(predictor);
  }


  void SetUpdater(std::unique_ptr<TUpdater>&& updater) {
    static_assert(TUpdater::InputsAtCompileTime == StateDim && TUpdater::ValuesAtCompileTime == MeasDim,
            "Updater dimensions do not match the Kalman filter dimensions");
    static_assert(std::is_same<typename TUpdater::ScalarType, T>::value, 
            "Updater scalar type does not match the Kalman filter type.");
    updater_ = std::move(updater);
  }
  

  const StateVec& State() const { return x_; }
  const StateMat& Uncertainty() const { return P_; }

  const StateMat& ProcessNoise() const { return Q_;}
  const MeasMat& MeasurementNoise() const { return R_; }

 private:
  StateVec x_ = StateVec::Zero();
  StateMat P_ = StateMat::Zero();

  StateMat Q_;
  MeasMat R_;

  std::unique_ptr<TPredictor> predictor_;
  std::unique_ptr<TUpdater> updater_;
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H