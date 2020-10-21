#ifndef KFCPP_KALMAN_FILTER
#define KFCPP_KALMAN_FILTER

#include <eigen3/Eigen/Core>

namespace KalmanCpp {

class KalmanFilter {
 public:
  KalmanFilter(size_t state_dim);
  ~KalmanFilter() = default;

 private:
  Eigen::VectorXf state_;
  Eigen::MatrixXf cov_;
};

}  // namespace KalmanCpp

#endif  // KFCPP_KALMAN_FILTER