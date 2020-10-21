#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>

namespace KalmanCpp {

class ExtendedKalmanFilter {
 public:
  ExtendedKalmanFilter();
  ~ExtendedKalmanFilter();

  void Predict();
  void Update(const Eigen::VectorXf& z);

 private:
  Eigen::VectorXf x_;
  Eigen::MatrixXf P_;

  Eigen::MatrixXf Q_;
  Eigen::MatrixXf R_;
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H