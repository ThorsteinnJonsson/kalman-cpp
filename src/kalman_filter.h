#ifndef KFCPP_KALMAN_FILTER
#define KFCPP_KALMAN_FILTER

#include <eigen3/Eigen/Core>

namespace KalmanCpp {

class KalmanFilter {
 public:
  KalmanFilter(size_t state_dim, size_t meas_dim);
  ~KalmanFilter() = default;

  void Predict();
  void Update(const Eigen::VectorXf& z);

  void InitState(const Eigen::VectorXf& state, const Eigen::MatrixXf& cov);
  void InitUncertainty(const Eigen::MatrixXf& process_noise, const Eigen::MatrixXf& measurement_noise);
  void SetStateTransitionMatrix(const Eigen::MatrixXf& state_transition_matrix);
  void SetMeasurementFunctionMatrix(const Eigen::MatrixXf& meas_function_matrix);

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

  Eigen::MatrixXf F_;
  Eigen::MatrixXf H_;

};

}  // namespace KalmanCpp

#endif  // KFCPP_KALMAN_FILTER