#ifndef KALMAN_CPP_EXAMPLES_COMMON_H
#define KALMAN_CPP_EXAMPLES_COMMON_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#include <matplot/matplot.h>
#include <eigen3/Eigen/Core>

struct Measurement {
  float ground_truth;
  float value;
  float timestamp;
};


#endif  // KALMAN_CPP_EXAMPLES_COMMON_H