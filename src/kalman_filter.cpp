#include "kalman_filter.h"

using namespace KalmanCpp;

KalmanFilter::KalmanFilter(size_t state_dim) {
  state_ = Eigen::Vector2f::Zero(state_dim);
  cov_ = Eigen::MatrixXf::Zero(state_dim,state_dim);
}
