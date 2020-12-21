#include "extended_kalman_filter.h"
#include "examples_common.h"
#include "ball_simulator.h"


void PlotResult(const std::vector<Eigen::Vector2f>& measurements,
                const std::vector<Eigen::Vector2f>& estimates) {
  std::vector<double> x_meas;
  std::vector<double> y_meas;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(x_meas),
                 [](const Eigen::Vector2f& m) { return m(0); });
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(y_meas),
                 [](const Eigen::Vector2f& m) { return m(1); });

  matplot::plot(x_meas, y_meas, "o")->marker_size(6);
  matplot::hold(matplot::on);


  std::vector<double> x_est;
  std::vector<double> y_est;
  std::transform(estimates.begin(),
                 estimates.end(),
                 std::back_inserter(x_est),
                 [](const Eigen::Vector2f& m) { return m(0); });
  std::transform(estimates.begin(),
                 estimates.end(),
                 std::back_inserter(y_est),
                 [](const Eigen::Vector2f& m) { return m(1); });

  matplot::plot(x_est, y_est, "-")->line_width(4);
  matplot::hold(matplot::on);


  matplot::axis(matplot::equal);
  matplot::show();
}


struct MyPredictor : public KalmanCpp::BasePredictor<MyPredictor, float, 4, KalmanCpp::JacobianMethod::Numerical> {
  
  template <typename InMat, typename OutMat>
  void GetPrediction(const InMat& in, OutMat& out) const {
    out = OutMat::Zero();
    out(0) = in(0) + this->Timestep() * out(1);
    out(1) = in(1);
    out(2) = in(2) + this->Timestep() * out(3);
    out(3) = in(3);
  }

  template <typename ControlMat>
  void GetControlInput(const ControlMat& control_input, ControlMat& control_out) {
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(4,4); // Control model matrix
    B(3,3) = 1.0f;
    control_out = B * control_input;
  }

};

struct MyUpdater : public KalmanCpp::BaseUpdater<MyUpdater, float, 4, 2, KalmanCpp::JacobianMethod::Numerical> {
  
  template <typename InMat, typename OutMat>
  void GetMeasurement(const InMat& in, OutMat& out) const {
    out = OutMat::Zero();
    out(0) = in(0);
    out(1) = in(2);
  }

};


template <typename T, int StateDim, int MeasDim, typename TPredictor, typename TUpdater>
KalmanCpp::ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater> SetupFilter() {
  KalmanCpp::ExtendedKalmanFilter<T, StateDim, MeasDim, TPredictor, TUpdater> kf;

  // Initial values
  const float initial_velocity = 50.0f;
  const float launch_angle = M_PI/180.0f * 35.0f;
  Eigen::Vector4f init_state = Eigen::Vector4f::Zero();  //[x vx y vy]
  init_state(0) = 0.0f;
  init_state(1) = initial_velocity * std::cos(launch_angle);
  init_state(2) = 1.0f;
  init_state(3) = initial_velocity * std::sin(launch_angle);
  Eigen::Matrix4f init_cov = Eigen::Matrix4f::Identity();
  kf.InitState(init_state, init_cov);

  // Noise
  Eigen::Matrix4f process_noise = Eigen::Matrix4f::Identity() * 0.1f;
  Eigen::Matrix2f measurement_noise = Eigen::Matrix2f::Identity() * 0.5f;
  kf.InitUncertainty(process_noise, measurement_noise);

  std::unique_ptr<MyPredictor> predictor = std::make_unique<MyPredictor>();
  kf.SetPredictor(std::move(predictor));

  std::unique_ptr<MyUpdater> updater = std::make_unique<MyUpdater>();
  kf.SetUpdater(std::move(updater));

  return kf;
}

void RunExample() {
  // Set up Kalman filter
  constexpr size_t state_dim = 4;
  constexpr size_t meas_dim = 2;
  auto kf = SetupFilter<float, state_dim, meas_dim, MyPredictor, MyUpdater>();

  // // Set up simulation
  const float gt_initial_speed = 50.0f;
  const float gt_launch_angle = M_PI/180.0f * 35.0f;
  Eigen::Vector2f initial_pos;
  initial_pos << 0.0f, 1.0f;
  Eigen::Vector2f initial_vel;
  initial_vel << gt_initial_speed * std::cos(gt_launch_angle),
                 gt_initial_speed * std::sin(gt_launch_angle);
  const Eigen::Vector2f sim_noise = Eigen::Vector2f::Constant(0.3f);
  BallSim ball(initial_pos, initial_vel, sim_noise);
  
  float t = 0.0f;
  const float dt = 0.1f;  // Assume constant timestep
  std::vector<Eigen::Vector2f> measurements;
  std::vector<Eigen::Vector2f> estimates;
  while (ball.AboveGround()) {
    t += dt;

    // Predict with control input
    Eigen::Vector4f u = Eigen::Vector4f::Zero();
    constexpr float g = 9.81f;
    u(3) = -g*dt;
    kf.Predict(u, dt);

    const float wind_vel = 0.0f;
    const Eigen::Vector2f measurement = ball.Update(dt, wind_vel);
    measurements.push_back(measurement);

    kf.Update(measurement);

    Eigen::Vector2f estimate;
    estimate << kf.State()(0), kf.State()(2);
    estimates.push_back(estimate);
  }

  PlotResult(measurements, estimates);
}


int main() {
  RunExample();
  return 0;
}
