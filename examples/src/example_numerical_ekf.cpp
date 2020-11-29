#include <iostream>

#include "extended_kalman_filter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <matplot/matplot.h>

struct Measurement {
  float ground_truth;
  float value;
  float timestamp;
};

std::vector<Measurement> GenerateMeasurements(float meas_var,
                                              float process_var,
                                              float dt = 1.0f,
                                              size_t count = 50) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> dist;

  float pos = 0.0f;
  float vel = 1.0f;

  float meas_std = std::sqrt(meas_var);
  float process_std = std::sqrt(process_var);

  std::vector<Measurement> measurements;
  measurements.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    Measurement m;
    float v = vel + dist(gen) * process_std;
    pos += v * dt;
    m.ground_truth = pos;
    m.value = pos + dist(gen) * meas_std;
    m.timestamp = dt * i;
    measurements.push_back(m);
  }
  return measurements;
}

void PlotResult(const std::vector<Measurement>& measurements,
                const std::vector<Eigen::VectorXf>& track,
                float true_meas_var) {
  std::vector<double> time;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(time),
                 [](const Measurement& m) { return m.timestamp; });
  std::vector<double> ground_truth;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(ground_truth),
                 [](const Measurement& m) { return m.ground_truth; });
  std::vector<double> upper_std;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(upper_std),
                 [&](const Measurement& m) {
                   return m.ground_truth + std::sqrt(true_meas_var);
                 });
  std::vector<double> lower_std;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(lower_std),
                 [&](const Measurement& m) {
                   return m.ground_truth - std::sqrt(true_meas_var);
                 });

  std::vector<double> measured_values;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(measured_values),
                 [](const Measurement& m) { return m.value; });

  std::vector<double> kf_result;
  std::transform(track.begin(),
                 track.end(),
                 std::back_inserter(kf_result),
                 [](const Eigen::VectorXf& t) { return t(0); });

  matplot::plot(time, ground_truth, "-")->line_width(4);
  matplot::hold(matplot::on);
  matplot::plot(time, upper_std, "--")->line_width(1).color("k");
  matplot::hold(matplot::on);
  matplot::plot(time, lower_std, "--")->line_width(1).color("k");
  matplot::hold(matplot::on);
  matplot::plot(time, measured_values, "x")->marker_size(6);
  matplot::hold(matplot::on);
  matplot::plot(time, kf_result, "-")->line_width(4);
  matplot::show();
}

template <typename T, int StateDim, int MeasDim, KalmanCpp::JacobianCalculationMethod JacobianMethod >
KalmanCpp::ExtendedKalmanFilter<float, StateDim, MeasDim, JacobianMethod> SetupFilter(
    float process_var, float meas_var, float dt) {
  KalmanCpp::ExtendedKalmanFilter<T, StateDim, MeasDim, JacobianMethod> kf;

  Eigen::Vector2f init_state =
      Eigen::Vector2f::Zero();  // Position and velocity
  Eigen::Matrix2f init_cov;
  init_cov << 500.0f, 0.0f, 0.0f, 49.0f;
  kf.InitState(init_state, init_cov);

  // Noise
  Eigen::Matrix2f process_noise;
  process_noise << 0.25f * std::pow(dt, 4) * process_var,
      0.5f * std::pow(dt, 3) * process_var,
      0.5f * std::pow(dt, 3) * process_var, std::pow(dt, 2) * process_var;
  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf(1, 1);
  measurement_noise << meas_var;
  kf.InitUncertainty(process_noise, measurement_noise);

  // State transition
  std::unique_ptr<KalmanCpp::Predictor<T,StateDim,JacobianMethod>> predictor = std::make_unique<KalmanCpp::Predictor<T,StateDim,JacobianMethod>>();
  kf.SetPredictor(std::move(predictor));

  // Measurment function
  auto measurement_func = [](const Eigen::VectorXf& x) -> Eigen::VectorXf {
    Eigen::MatrixXf z(1, 1);
    z << x(0);
    return z;
  };
  kf.SetMeasurementFunction(std::move(measurement_func));

  auto measurement_jacobian = [](const Eigen::VectorXf&) -> Eigen::MatrixXf {
    Eigen::MatrixXf jacobian = Eigen::MatrixXf::Zero(1, 2);
    jacobian(0, 0) = 1.0f;
    return jacobian;
  };
  kf.SetMeasurementJacobian(std::move(measurement_jacobian));

  return kf;
}

void RunExample() {
  // Set up Kalman filter
  constexpr size_t state_dim = 2;
  constexpr size_t meas_dim = 1;
  constexpr float filter_process_var = 0.1f;
  constexpr float filter_meas_var = 10.0f;
  constexpr float dt = 1.0f;  // Assume constant timestep
  auto kf =
      SetupFilter<float, state_dim, meas_dim, KalmanCpp::JacobianCalculationMethod::Numerical>(filter_process_var, filter_meas_var, dt);

  // Get measurements
  constexpr size_t num_meas = 50;
  constexpr float true_meas_var = 10.0f;
  constexpr float true_process_var = 0.1f;
  std::vector<Measurement> measurements =
      GenerateMeasurements(true_meas_var, true_process_var, dt, num_meas);

  // Run simulation
  std::vector<Eigen::VectorXf> track;
  for (auto&& meas : measurements) {
    kf.Predict(dt);

    Eigen::MatrixXf z(meas_dim, 1);
    z << meas.value;
    kf.NumericalUpdate(z);

    track.push_back(kf.State());
  }

  PlotResult(measurements, track, true_meas_var);
}

int main() {
  RunExample();
  return 0;
}
