#include "straight_line_measurements.h"

std::vector<Measurement> GenerateStraightLineMeasurements(float meas_var,
                                                          float process_var,
                                                          float dt,
                                                          size_t count) {
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