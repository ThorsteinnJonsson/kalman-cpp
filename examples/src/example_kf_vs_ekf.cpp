#include <iostream>

#include "extended_kalman_filter.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <iostream>

#include <matplot/matplot.h>

KalmanCpp::ExtendedKalmanFilter SetupFilter(size_t state_dim,
                                    size_t meas_dim) {

  KalmanCpp::ExtendedKalmanFilter kf(state_dim, meas_dim);

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

  // State transition
  auto state_transition = [](const Eigen::VectorXf& x, float timestep) -> Eigen::VectorXf {
    Eigen::Vector4f x_new;
    x_new(0) = x(0) + timestep * x(1);
    x_new(1) = x(1);
    x_new(2) = x(2) + timestep * x(3);
    x_new(3) = x(3);
    return x_new; 
  };
  kf.SetStateTransitionFunction(std::move(state_transition));

  auto state_transition_jacobian = [](const Eigen::VectorXf&, float timestep) -> Eigen::MatrixXf {
    Eigen::Matrix4f jacobian = Eigen::Matrix4f::Identity();
    jacobian(0,1) = timestep;
    jacobian(2,3) = timestep;
    return jacobian;
  };
  kf.SetStateTransitionJacobian(std::move(state_transition_jacobian));

  // Measurment function 
  auto measurement_func = [](const Eigen::VectorXf& x) -> Eigen::VectorXf {
    Eigen::Vector2f z;
    z(0) = x(0);
    z(1) = x(2);
    return z;
  };
  kf.SetMeasurementFunction(std::move(measurement_func));

  auto measurement_jacobian = [](const Eigen::VectorXf&) -> Eigen::MatrixXf {
    Eigen::MatrixXf jacobian = Eigen::MatrixXf::Zero(2,4);
    jacobian(0,0) = 1.0f;
    jacobian(1,2) = 1.0f;
    return jacobian;
  };
  kf.SetMeasurementJacobian(std::move(measurement_jacobian));

  // Control input
  Eigen::MatrixXf B = Eigen::MatrixXf::Zero(4,4);
  B(3,3) = 1.0f;
  kf.SetControlInputFunction(B);

  return kf;
}




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




class BallSim {
 public:
  BallSim(float x0, float y0, float v, float launch_angle, const Eigen::Vector2f& noise) 
      : x_(x0),
        y_(y0),
        vx_(v * std::cos(launch_angle)),
        vy_(v * std::sin(launch_angle)),
        noise_(noise) {
    std::random_device rd{};
    gen_ = std::mt19937{rd()};
  }

  Eigen::Vector2f Update(float dt, float vel_wind=0.0f) {
    x_ += vx_ * dt;
    y_ += vy_ * dt;

    const float vx_wind = vx_ - vel_wind;
    const float v = std::sqrt(vx_wind*vx_wind + vy_*vy_);
    const float F = DragForce(v);

    vx_ -= F * vx_wind * dt;
    vy_ += -9.81f * dt - F * vy_ * dt;

    Eigen::Vector2f pos;
    pos << x_ + RandFloat() * noise_(0),
           y_ + RandFloat() * noise_(1);
    return pos;
  }

  bool AboveGround() const { return y_ > 0.0f; }

 private:
  float RandFloat() { return dist_(gen_); }

  float DragForce(float velocity) {
    const float B_m = 0.0039f + 0.0058f / (1.f + std::exp((velocity-35.f)/5.f));
    return B_m * velocity;
  }

 private: 
  float x_;
  float y_;
  float vx_;
  float vy_;
  const Eigen::Vector2f noise_;

  std::mt19937 gen_;
  std::normal_distribution<float> dist_;
};




void RunExample() {
  // Set up Kalman filter
  const size_t state_dim = 4;
  const size_t meas_dim = 2;
  KalmanCpp::ExtendedKalmanFilter kf = SetupFilter(state_dim, meas_dim);

  // Set up simulation
  const float x0 = 0.0f;
  const float y0 = 1.0f;
  const float initial_velocity = 50.0f;
  const float launch_angle = M_PI/180.0f * 35.0f;
  const Eigen::Vector2f sim_noise = Eigen::Vector2f::Constant(0.3f);
  BallSim ball(x0, y0, initial_velocity, launch_angle, sim_noise);
  
  float t = 0.0f;
  const float dt = 0.1f;  // Assume constant timestep
  std::vector<Eigen::Vector2f> measurements;
  std::vector<Eigen::Vector2f> estimates;
  while (ball.AboveGround()) {
    t += dt;

    // Predict with control inptu
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
