#ifndef KFCPP_KALMAN_FILTER
#define KFCPP_KALMAN_FILTER

#include <eigen3/Eigen/Core>

namespace KalmanCpp {

class KalmanFilter {
 public:
  KalmanFilter(size_t state_dim, size_t meas_dim);
  ~KalmanFilter() = default;

  /* Equations
  x = F*x + Bu
  P = F*P*F_t + Q_
  */
  void Predict();

  /* Equations
  y = z - H*x
  K = P*H_t*(H*P*H_t + R)^-1
  x = x + Ky
  P = (I - K*H)P  or  P = (I-K*H)*P*(I-K*H)_t + K*R*K_t
  */
  void Update(const Eigen::VectorXf& z);


  void InitState(const Eigen::VectorXf& state, const Eigen::MatrixXf& cov);
  void InitUncertainty(const Eigen::MatrixXf& process_noise, const Eigen::MatrixXf& measurement_noise);
  void SetStateTransitionMatrix(const Eigen::MatrixXf& state_transition_matrix);
  void SetMeasurementFunctionMatrix(const Eigen::MatrixXf& meas_function_matrix);

  const Eigen::VectorXf& State() const { return x_; }

 private:
  const Eigen::Index nx_;
  const Eigen::Index nz_;

  Eigen::VectorXf x_;
  Eigen::MatrixXf P_;

  Eigen::MatrixXf Q_;
  Eigen::MatrixXf R_;

  Eigen::MatrixXf F_;
  Eigen::MatrixXf H_;

};

}  // namespace KalmanCpp

#endif  // KFCPP_KALMAN_FILTER