#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>
#include <functional>

namespace KalmanCpp {

class ExtendedKalmanFilter {
 public:
  ExtendedKalmanFilter(size_t state_dim, size_t meas_dim);
  ~ExtendedKalmanFilter() = default;

  void Predict(float dt);
  void Predict(const Eigen::VectorXf& u, float dt);
  void Update(const Eigen::VectorXf& z);

  void InitState(const Eigen::VectorXf& state, const Eigen::MatrixXf& cov);
  void InitUncertainty(const Eigen::MatrixXf& process_noise, const Eigen::MatrixXf& measurement_noise);

  void SetStateTransitionFunction(std::function<Eigen::VectorXf(const Eigen::VectorXf&, float dt)>&& func) { f_ = func; }
  void SetStateTransitionJacobian(std::function<Eigen::MatrixXf(const Eigen::VectorXf&, float dt)>&& func) { f_jacobian_ = func; }

  void SetMeasurementFunction(std::function<Eigen::VectorXf(const Eigen::VectorXf&)>&& func) { h_ = func; }
  void SetMeasurementJacobian(std::function<Eigen::MatrixXf(const Eigen::VectorXf&)>&& func) { h_jacobian_ = func; }

  void SetControlInputFunction(const Eigen::MatrixXf& B) { B_ = B; }

  const Eigen::VectorXf& State() const { return x_; }
  const Eigen::MatrixXf& Uncertainty() const { return P_; }

  void Print() const;

 private:
  size_t nx_;
  size_t nz_;

  Eigen::VectorXf x_;
  Eigen::MatrixXf P_;

  Eigen::MatrixXf Q_;
  Eigen::MatrixXf R_;

  std::function<Eigen::VectorXf(const Eigen::VectorXf&, float dt)> f_; // convert state to next state
  std::function<Eigen::MatrixXf(const Eigen::VectorXf&, float dt)> f_jacobian_;
  
  std::function<Eigen::VectorXf(const Eigen::VectorXf&)> h_; // convert state to measurement
  std::function<Eigen::MatrixXf(const Eigen::VectorXf&)> h_jacobian_;

  Eigen::MatrixXf B_;

};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H