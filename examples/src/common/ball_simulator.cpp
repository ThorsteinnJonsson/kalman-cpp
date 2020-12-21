#include "ball_simulator.h"

BallSim::BallSim(const Eigen::Vector2f& initial_pos,
                 const Eigen::Vector2f& initial_vel,
                 const Eigen::Vector2f& noise)
    : pos_(initial_pos), vel_(initial_vel), noise_(noise) {
  std::random_device rd{};
  gen_ = std::mt19937{rd()};
}

Eigen::Vector2f BallSim::Update(float dt, float vel_wind) {
  pos_ += vel_ * dt;

  const float vx_wind = vel_(0) - vel_wind;
  const float v = std::sqrt(vx_wind * vx_wind + vel_(1) * vel_(1));
  const float F = DragForce(v);

  vel_(0) -= F * vx_wind * dt;
  vel_(1) += -9.81f * dt - F * vel_(1) * dt;

  Eigen::Vector2f measurement;
  measurement << pos_(0) + RandFloat() * noise_(0), pos_(1) + RandFloat() * noise_(1);
  return measurement;
}