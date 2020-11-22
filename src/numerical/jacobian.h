#ifndef KALMAN_CPP_NUMERICAL_JACOBIAN_H
#define KALMAN_CPP_NUMERICAL_JACOBIAN_H

#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/AutoDiff>

namespace KalmanCpp::Numerical {

template <typename T, int N>
using JacobianFunction = std::function<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>(
    const std::array<Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>, N>&, float)>;

template <typename T, int N>
using JacobianScalar = Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>;

template <typename T, int N, int M>
Eigen::Matrix<T, N, M> CalculateJacobian(const Eigen::Matrix<T, N, 1>& x,
                                         std::array<JacobianFunction<T, N>, M> functions,
                                         float timestep) {
  Eigen::Matrix<T, N, M> jacobian;

  for (int i = 0; i < M; ++i) {
    std::array<JacobianScalar<T, N>, N> scalars;
    for (int j = 0; j < N; ++j) {
      scalars[j].value() = x(j);
      scalars[j].derivatives() = Eigen::Matrix<T, N, 1>::Unit(N, j);
    }

    const JacobianFunction<T, N>& func = functions[i];
    JacobianScalar<T, N> F = func(scalars, timestep);

    for (int j = 0; j < N; ++j) {
      jacobian(i, j) = F.derivatives()(j);
    }
  }

  return jacobian;
}

}  // namespace KalmanCpp::Numerical

#endif  // KALMAN_CPP_NUMERICAL_JACOBIAN_H