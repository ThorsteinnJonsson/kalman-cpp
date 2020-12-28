#ifndef KALMAN_CPP_EXAMPLE_COMMON_PLOTTING_H
#define KALMAN_CPP_EXAMPLE_COMMON_PLOTTING_H

#include "examples_common.h"

void PlotStraightLineExample(const std::vector<Measurement>& measurements,
                             const std::vector<Eigen::VectorXf>& track,
                             float true_meas_var);

void PlotBallSim(const std::vector<Eigen::Vector2f>& measurements,
                const std::vector<Eigen::Vector2f>& estimates);

#endif  // KALMAN_CPP_EXAMPLE_COMMON_PLOTTING_H