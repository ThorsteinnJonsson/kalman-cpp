#ifndef KALMAN_CPP_NUMERICAL_JACOBIAN_H
#define KALMAN_CPP_NUMERICAL_JACOBIAN_H

#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace KalmanCpp::Numerical {

template <typename T, int N, int M>
Eigen::Matrix<T, N, M> CalculateJacobian(const Eigen::Matrix<T, N, 1>& x,
    std::array<std::function<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>(const std::array<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>, N>&, float)>, M> fs, float dt) {
  Eigen::Matrix<T, N, M> jacobian;

  for (int i = 0; i < M; ++i) {
    std::array<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>, N> scalars;
    for (int j = 0; j < N; ++j) {
      scalars[j].value() = x(j);
      scalars[j].derivatives() = Eigen::Matrix<T, N, 1>::Unit(N,j);
    }

    std::function<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>(const std::array<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>, N>&, float)> func = fs[i];
    Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>> F = func(scalars, dt); 

    for (int j = 0; j < N; ++j) {
      jacobian(i,j) = F.derivatives()(j);
    }
  }
  



  return jacobian;
}

}  // namespace KalmanCpp::Numerical

#endif  // KALMAN_CPP_NUMERICAL_JACOBIAN_H