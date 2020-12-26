#include <gtest/gtest.h>
#include "extended_kalman_filter.h"

using namespace KalmanCpp;

class TestEkfPrediction : public ::testing::Test {
 protected:
  // State is [x_pos, x_vel]
  // Measurement is [x_pos]
  static constexpr int StateDim = 2;
  static constexpr int MeasDim = 1;

  struct TestPredictor : public KalmanCpp::BasePredictor<TestPredictor, float, StateDim, KalmanCpp::JacobianMethod::Numerical> {
    template <typename InMat, typename OutMat>
    void GetPrediction(const InMat& in, OutMat& out) const {
      out = OutMat::Zero();
      out(0) = in(0) + in(1) * this->Timestep();
      out(1) = in(1);
    }
  };

  struct TestUpdater : public KalmanCpp::BaseUpdater<TestUpdater, float, StateDim, MeasDim, KalmanCpp::JacobianMethod::Numerical> {
    template <typename InMat, typename OutMat>
    void GetMeasurement(const InMat& in, OutMat& out) const {
      out = OutMat::Zero();
    }
  };

  using EkfType = ExtendedKalmanFilter<float, StateDim, MeasDim, TestPredictor, TestUpdater>;

  std::unique_ptr<EkfType> ekf_;


  Eigen::Matrix<float,StateDim,StateDim> true_jacobian_;
  Eigen::Matrix<float,StateDim,StateDim> true_process_noise_;
  Eigen::Matrix<float,StateDim,StateDim> expected_cov_;

  void SetUp() override {
    ekf_ = std::make_unique<EkfType>();

    Eigen::Vector2f init_state;
    init_state << 0.0f, 5.0f;
    Eigen::Matrix2f init_cov;
    init_cov << 10.0f, 0.0f, 0.0f, 10.0f;
    ekf_->InitState(init_state, init_cov);
    
    // Noise
    Eigen::Matrix2f process_noise = Eigen::Matrix2f::Identity();
    Eigen::Matrix<float,1,1> measurement_noise = Eigen::Matrix<float,1,1>::Identity();
    ekf_->InitUncertainty(process_noise, measurement_noise);

    ekf_->SetPredictor(std::make_unique<TestPredictor>());

    // These are ground truth values
    true_process_noise_ = process_noise;
    true_jacobian_ = Eigen::Matrix<float,StateDim,StateDim>::Identity();
    true_jacobian_(0,1) = 1.0f;
    expected_cov_ = init_cov;
  }

  void TearDown() override {}

  static constexpr int num_iterations_ = 10;
};


TEST_F(TestEkfPrediction, TestPredictionState) {

  float expected_pos = 0.0f;
  const float expected_vel = 5.0f;
  
  const auto& initial_state = ekf_->State();
  EXPECT_NEAR(initial_state(0), expected_pos, 1e-6);
  EXPECT_NEAR(initial_state(1), expected_vel, 1e-6);

  const float dt = 1.0f;
  for (int i = 0; i < num_iterations_; ++i) {
    expected_pos += expected_vel * dt;

    ekf_->Predict(dt);
    const auto& state = ekf_->State();

    EXPECT_NEAR(state(0), expected_pos, 1e-6);
    EXPECT_NEAR(state(1), expected_vel, 1e-6);
  }
}

TEST_F(TestEkfPrediction, TestPredictionCov) {

  const auto& initial_cov = ekf_->Uncertainty();
  EXPECT_TRUE((initial_cov-expected_cov_).isZero(1e-9));

  const float dt = 1.0f;
  for (int i = 0; i < num_iterations_; ++i) {

    expected_cov_ = true_jacobian_ * expected_cov_ * true_jacobian_.transpose() + true_process_noise_;

    ekf_->Predict(dt);
    const auto& cov = ekf_->Uncertainty();

    EXPECT_TRUE((cov-expected_cov_).isZero(1e-9));
  }
}