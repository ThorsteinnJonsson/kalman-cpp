#include "kalman_filter.h"

#include <iostream>

using namespace KalmanCpp;

KalmanFilter::KalmanFilter() {
  state_ = Eigen::Vector2f::Zero();
  cov_ = Eigen::MatrixXf::Constant(5,5,42);
}

void KalmanFilter::SayHello() const {
  std::cout << "Hello World!\n";
  
  std::cout << "state:" << std::endl;
  std::cout << state_ << std::endl;
  
  std::cout << "cov:" << std::endl;
  std::cout << cov_ << std::endl;
}
