#include "example_plotting.h"

void PlotStraightLineExample(const std::vector<Measurement>& measurements,
                             const std::vector<Eigen::VectorXf>& track,
                             float true_meas_var) {
  std::vector<double> time;
  std::transform(
      measurements.begin(), measurements.end(), std::back_inserter(time), [](const Measurement& m) {
        return m.timestamp;
      });
  std::vector<double> ground_truth;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(ground_truth),
                 [](const Measurement& m) { return m.ground_truth; });
  std::vector<double> upper_std;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(upper_std),
                 [&](const Measurement& m) { return m.ground_truth + std::sqrt(true_meas_var); });
  std::vector<double> lower_std;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(lower_std),
                 [&](const Measurement& m) { return m.ground_truth - std::sqrt(true_meas_var); });

  std::vector<double> measured_values;
  std::transform(measurements.begin(),
                 measurements.end(),
                 std::back_inserter(measured_values),
                 [](const Measurement& m) { return m.value; });

  std::vector<double> kf_result;
  std::transform(
      track.begin(), track.end(), std::back_inserter(kf_result), [](const Eigen::VectorXf& t) {
        return t(0);
      });

  matplot::plot(time, ground_truth, "-")->line_width(4);
  matplot::hold(matplot::on);
  matplot::plot(time, upper_std, "--")->line_width(1).color("k");
  matplot::hold(matplot::on);
  matplot::plot(time, lower_std, "--")->line_width(1).color("k");
  matplot::hold(matplot::on);
  matplot::plot(time, measured_values, "x")->marker_size(6);
  matplot::hold(matplot::on);
  matplot::plot(time, kf_result, "-")->line_width(4);
  matplot::show();
}

void PlotBallSim(const std::vector<Eigen::Vector2f>& measurements,
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
  std::transform(
      estimates.begin(), estimates.end(), std::back_inserter(x_est), [](const Eigen::Vector2f& m) {
        return m(0);
      });
  std::transform(
      estimates.begin(), estimates.end(), std::back_inserter(y_est), [](const Eigen::Vector2f& m) {
        return m(1);
      });

  matplot::plot(x_est, y_est, "-")->line_width(4);
  matplot::hold(matplot::on);

  matplot::axis(matplot::equal);
  matplot::show();
}