#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/AutoDiff>
#include <iostream>

// TODO also look at
// https://joelcfd.com/automatic-differentiation/

// template <typename T>
// T myfun(T const & a, T const & b) {
//   T c = pow(sin(a),2.) + pow(cos(b),2.) + 1.;
// 	return c;
// }


int main() {
  typedef Eigen::AutoDiffScalar<Eigen::VectorXd> AScalar;

  auto myfun = [](const auto& a, const auto& b) {
    return pow(sin(a),2.) + pow(cos(b),2.) + 1.;
  };

  AScalar A, B;
  A.value() = 2.0;
  B.value() = 3.0;

  A.derivatives() = Eigen::VectorXd::Unit(2,0);
  B.derivatives() = Eigen::VectorXd::Unit(2,1);

  AScalar C = myfun(A,B);

  std::cout << "Result: " << myfun(2.0, 3.0) << std::endl;

  //
  // f(x,y) = sin(x)^2 + cos(y)^2 + 1
  //
  std::cout << "Result: " << C.value() << std::endl;

  //
  // [df/dx df/xy] = [2cos(x)sin(x), -2cos(y)sin(y)]
  //
  std::cout << "Gradient: " << C.derivatives().transpose() << std::endl;

  return 0;
}