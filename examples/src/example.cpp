#include <iostream>

#include "kalman_filter.h"

#include <chrono>
#include <random>

struct Measurement {
  float val;
  float actual;
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
    m.actual = pos;
    m.val = pos + dist(gen) * meas_std;
    measurements.push_back(m);
  }
  return measurements;
}

void RunExample() {
  // Set up Kalman filter
  const size_t state_dim = 2;
  const size_t meas_dim = 1;
  KalmanCpp::KalmanFilter kf(state_dim, meas_dim);

  // Initial values
  Eigen::Vector2f init_state =
      Eigen::Vector2f::Zero();  // Position and velocity
  Eigen::Matrix2f init_cov;
  init_cov << 500.0f, 0.0f, 0.0f, 49.0f;
  kf.InitState(init_state, init_cov);

  // Noise
  Eigen::Matrix2f process_noise = Eigen::Matrix2f::Identity();
  process_noise(1, 1) = 0.001f;
  Eigen::MatrixXf measurement_noise(1, 1);
  measurement_noise << 5.0f;
  kf.InitUncertainty(process_noise, measurement_noise);

  // State transition
  const float dt = 0.1f;
  Eigen::Matrix2f state_transition_matrix = Eigen::Matrix2f::Identity();
  state_transition_matrix(0, 1) = dt;
  kf.SetStateTransitionMatrix(state_transition_matrix);

  // Measurment function
  Eigen::MatrixXf measurement_function =
      Eigen::MatrixXf::Zero(state_dim, meas_dim);
  measurement_function(0, 0) = 1.0f;
  kf.SetMeasurementFunctionMatrix(measurement_function);

  // Get measurements
  std::vector<Measurement> measurements =
      GenerateMeasurements(5.0f, 0.1f, 0.1f, 50);

  // Run simulation
  std::vector<Eigen::VectorXf> track;
  for (auto&& meas : measurements) {
    kf.Predict();

    Eigen::MatrixXf z(1, 1);
    z << meas.actual;
    kf.Update(z);

    track.push_back(kf.State());
    std::cout << meas.actual << "  |  " << meas.val << "\n";
    std::cout << track.back().transpose() << "\n---\n";
  }
}

int main() {
  RunExample();
  return 0;
}
