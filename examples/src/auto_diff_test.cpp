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

void AutoDiffScalarExample() {
  typedef Eigen::AutoDiffScalar<Eigen::VectorXd> AScalar;

  auto myfun = [](const auto& a, const auto& b) { return pow(sin(a), 2.) + pow(cos(b), 2.) + 1.; };

  AScalar A, B;
  A.value() = 2.0;
  B.value() = 3.0;

  A.derivatives() = Eigen::VectorXd::Unit(2, 0);
  B.derivatives() = Eigen::VectorXd::Unit(2, 1);

  AScalar C = myfun(A, B);

  std::cout
      << "### AutoDiffScalarExample ##########################################################\n";
  std::cout << "f(x,y) = sin(x)^2 + cos(y)^2 + 1" << std::endl;
  std::cout << "f(2,3): " << myfun(2.0, 3.0) << " [from function]\n";

  //
  // f(x,y) = sin(x)^2 + cos(y)^2 + 1
  //
  std::cout << "f(2,3): " << C.value() << " [from Eigen Autodiff]\n";

  //
  // [df/dx df/xy] = [2cos(x)sin(x), -2cos(y)sin(y)]
  //
  std::cout << "[df/dx df/xy] = [2cos(x)sin(x), -2cos(y)sin(y)] at (2,3):" << std::endl;
  std::cout << "Gradient: " << C.derivatives().transpose() << std::endl;
  std::cout
      << "####################################################################################\n";
}

template <typename T, int N_IN, int N_OUT>
struct AutoDiffFunctor {
  typedef Eigen::Matrix<T, N_IN, 1> InputType;
  typedef Eigen::Matrix<T, N_OUT, 1> ValueType;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };

  template <typename T1, typename T2>
  void operator()(const T1& input, T2* output) const {
    (*output)(0, 0) = pow(sin(input(0, 0)), 2.) + pow(cos(input(1, 0)), 2.) + 1.0f;
  };
};

void AutoDiffJacobianExample() {
  Eigen::Matrix<float, 2, 1> in = {2, 3};
  Eigen::Matrix<float, 1, 1> out;
  Eigen::Matrix<float, 1, 2> jacobian;

  Eigen::AutoDiffJacobian<AutoDiffFunctor<float, 2, 1>> auto_differ;
  auto_differ(in, &out, &jacobian);

  std::cout
      << "### AutoDiffJacobianExample ########################################################\n";
  std::cout << "f(x,y) = sin(x)^2 + cos(y)^2 + 1" << std::endl;

  std::cout << "x:\n" << in.transpose() << "\n";
  std::cout << "f(x):\n" << out << "\n";
  std::cout << "[df/dx df/xy] = [2cos(x)sin(x), -2cos(y)sin(y)] at x:" << std::endl;
  std::cout << jacobian << "\n";

  std::cout
      << "####################################################################################\n";
}

int main() {
  AutoDiffScalarExample();
  AutoDiffJacobianExample();

  return 0;
}