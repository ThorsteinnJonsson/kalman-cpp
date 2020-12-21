#ifndef KALMAN_CPP_EXAMPLES_COMMON_BALL_SIMILATOR_H
#define KALMAN_CPP_EXAMPLES_COMMON_BALL_SIMILATOR_H

#include "extended_kalman_filter.h"

#include "examples_common.h"

class BallSim {
 public:
  BallSim(const Eigen::Vector2f& initial_pos,
          const Eigen::Vector2f& initial_vel,
          const Eigen::Vector2f& noise);

  Eigen::Vector2f Update(float dt, float vel_wind = 0.0f);

  inline bool AboveGround() const { return pos_(1) > 0.0f; }

 private:
  inline float RandFloat() { return dist_(gen_); }

  inline float DragForce(float velocity) {
    const float B_m = 0.0039f + 0.0058f / (1.f + std::exp((velocity - 35.f) / 5.f));
    return B_m * velocity;
  }

 private:
  Eigen::Vector2f pos_;
  Eigen::Vector2f vel_;
  const Eigen::Vector2f noise_;

  std::mt19937 gen_;
  std::normal_distribution<float> dist_;
};

#endif  // KALMAN_CPP_EXAMPLES_COMMON_BALL_SIMILATOR_H