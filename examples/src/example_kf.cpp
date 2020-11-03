#include <iostream>

#include "kalman_filter.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

#include <matplot/matplot.h>

KalmanCpp::KalmanFilter SetupFilter(size_t state_dim,
                                    size_t meas_dim,
                                    float meas_var,
                                    float process_var,
                                    float dt) {
  KalmanCpp::KalmanFilter kf(state_dim, meas_dim);

  // Initial values
  Eigen::Vector2f init_state =
      Eigen::Vector2f::Zero();  // Position and velocity
  Eigen::Matrix2f init_cov;
  init_cov << 500.0f, 0.0f, 0.0f, 49.0f;
  kf.InitState(init_state, init_cov);

  // Noise
  Eigen::Matrix2f process_noise;
  process_noise << 0.25f * std::pow(dt,4) * process_var,
                   0.5f * std::pow(dt,3) * process_var,
                   0.5f * std::pow(dt,3) * process_var,
                         std::pow(dt,2) * process_var;
  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf(1, 1);
  measurement_noise << meas_var;
  kf.InitUncertainty(process_noise, measurement_noise);

  // State transition
  Eigen::Matrix2f state_transition_matrix = Eigen::Matrix2f::Identity();
  state_transition_matrix(0, 1) = dt;
  kf.SetStateTransitionMatrix(state_transition_matrix);

  // Measurment function
  Eigen::MatrixXf measurement_function =
      Eigen::MatrixXf::Zero(meas_dim, state_dim);
  measurement_function(0, 0) = 1.0f;
  kf.SetMeasurementFunctionMatrix(measurement_function);

  return kf;
}

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
  std::transform(
      measurements.begin(),
      measurements.end(),
      std::back_inserter(upper_std),
      [&](const Measurement& m) { return m.ground_truth + std::sqrt(true_meas_var); });
  std::vector<double> lower_std;
  std::transform(
      measurements.begin(),
      measurements.end(),
      std::back_inserter(lower_std),
      [&](const Measurement& m) { return m.ground_truth - std::sqrt(true_meas_var); });

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

void RunExample() {
  // Set up Kalman filter
  const size_t state_dim = 2;
  const size_t meas_dim = 1;
  const float dt = 1.0f;  // Assume constant timestep
  const float meas_var = 10.0f;
  const float process_var = 0.1f;
  KalmanCpp::KalmanFilter kf = SetupFilter(state_dim, meas_dim, meas_var, process_var, dt);
  kf.Print();

  // Get measurements
  const size_t num_measurements = 50;
  const float true_meas_var = 10.0f;
  const float true_process_var = 0.1f;
  std::vector<Measurement> measurements =
      GenerateMeasurements(true_meas_var, true_process_var, dt, num_measurements);

  // Run simulation
  std::vector<Eigen::VectorXf> track;
  for (auto&& meas : measurements) {
    kf.Predict();

    Eigen::MatrixXf z(1, 1);
    z << meas.value;
    kf.Update(z);

    track.push_back(kf.State());
  }
  kf.Print();
  PlotResult(measurements, track, true_meas_var);
}


int main() {
  RunExample();
  return 0;
}
