#ifndef KALMAN_CPP_EKF_H
#define KALMAN_CPP_EKF_H

#include <eigen3/Eigen/Core>

namespace KalmanCpp {

class ExtendedKalmanFilter {
 public:
  ExtendedKalmanFilter();
  ~ExtendedKalmanFilter();

  void Predict();
  void Update();

 private:
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_EKF_H