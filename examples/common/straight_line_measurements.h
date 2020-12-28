#ifndef KALMAN_CPP_EXAMPLES_COMMON_STRAIGHT_LINE_MEASUREMENTS_H
#define KALMAN_CPP_EXAMPLES_COMMON_STRAIGHT_LINE_MEASUREMENTS_H

#include "examples_common.h"

std::vector<Measurement> GenerateStraightLineMeasurements(float meas_var,
                                                          float process_var,
                                                          float dt = 1.0f,
                                                          size_t count = 50);

#endif  // KALMAN_CPP_EXAMPLES_COMMON_STRAIGHT_LINE_MEASUREMENTS_H