#include "extended_kalman_filter.h"
#include "examples_common.h"
#include "straight_line_measurements.h"
#include "example_plotting.h"


// Define a predictor for the EKF. GetPrediction always has to be defined. GetJacobian only has to be defined if
// the Jacobian method is set to "analytical".
struct MyPredictor : public KalmanCpp::BasePredictor<MyPredictor, float, 2, KalmanCpp::JacobianMethod::Numerical> {
  
  template <typename InMat, typename OutMat>
  void GetPrediction(const InMat& in, OutMat& out) const {
    out = OutMat::Zero();
    out(0) = in(0) + in(1) * this->Timestep();
    out(1) = in(1);
  }

};

// Define a updater for the EKF. GetMeasurement always has to be defined. GetJacobian only has to be defined if
// the Jacobian method is set to "analytical".
struct MyUpdater : public KalmanCpp::BaseUpdater<MyUpdater, float, 2, 1, KalmanCpp::JacobianMethod::Numerical> {
  
  template <typename InMat, typename OutMat>
  void GetMeasurement(const InMat& in, OutMat& out) const {
    out = OutMat::Zero();
    out(0) = in(0);
  }

};


template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
KalmanCpp::ExtendedKalmanFilter<float, StateDim, MeasDim, TPredictor, TUpdater> SetupFilter(
    float process_var, float meas_var, float dt) {
  KalmanCpp::ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater> kf;

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

  std::unique_ptr<MyPredictor> predictor = std::make_unique<MyPredictor>();
  kf.SetPredictor(std::move(predictor));

  std::unique_ptr<MyUpdater> updater = std::make_unique<MyUpdater>();
  kf.SetUpdater(std::move(updater));

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
      SetupFilter<float, state_dim, meas_dim, MyPredictor, MyUpdater>(filter_process_var, filter_meas_var, dt);

  // Get measurements
  constexpr size_t num_meas = 50;
  constexpr float true_meas_var = 10.0f;
  constexpr float true_process_var = 0.1f;
  std::vector<Measurement> measurements =
      GenerateStraightLineMeasurements(true_meas_var, true_process_var, dt, num_meas);

  // Run simulation
  std::vector<Eigen::VectorXf> track;
  for (auto&& meas : measurements) {
    kf.Predict(dt);

    Eigen::MatrixXf z(meas_dim, 1);
    z << meas.value;
    kf.Update(z);

    track.push_back(kf.State());
  }

  PlotStraightLineExample(measurements, track, true_meas_var);
}

int main() {
  RunExample();
  return 0;
}
