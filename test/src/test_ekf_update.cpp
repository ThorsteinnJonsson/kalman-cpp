#include <gtest/gtest.h>
#include "extended_kalman_filter.h"

#include <random>

using namespace KalmanCpp;

class TestEkfUpdate : public ::testing::Test {
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
      out(0) = in(0);
    }
  };

  using EkfType = ExtendedKalmanFilter<float, StateDim, MeasDim, TestPredictor, TestUpdater>;

  std::unique_ptr<EkfType> ekf_;


  Eigen::Matrix<float,StateDim,StateDim> true_jacobian_;
  Eigen::Matrix<float,StateDim,StateDim> true_process_noise_;
  Eigen::Matrix<float,StateDim,StateDim> expected_cov_;
  const float measurement_noise_ = 0.01f;

  void SetUp() override {
    ekf_ = std::make_unique<EkfType>();

    Eigen::Vector2f init_state;
    init_state << 0.0f, 5.0f;
    Eigen::Matrix2f init_cov;
    init_cov << 10.0f, 0.0f, 0.0f, 10.0f;
    ekf_->InitState(init_state, init_cov);
    
    // Noise
    Eigen::Matrix2f process_noise = Eigen::Matrix2f::Identity();
    Eigen::Matrix<float,1,1> measurement_noise = Eigen::Matrix<float,1,1>::Constant(measurement_noise_);
    ekf_->InitUncertainty(process_noise, measurement_noise);

    ekf_->SetPredictor(std::make_unique<TestPredictor>());
    ekf_->SetUpdater(std::make_unique<TestUpdater>());

    // These are ground truth values
    true_process_noise_ = process_noise;
    true_jacobian_ = Eigen::Matrix<float,StateDim,StateDim>::Identity();
    true_jacobian_(0,1) = 1.0f;
    expected_cov_ = init_cov;
  }

  void TearDown() override {}

  float GenerateMeasurement(float true_pos) {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<float> dist;
    
    const float generated_pos = true_pos + dist(gen) * measurement_noise_;
    return generated_pos;
  }

  static constexpr int num_iterations_ = 10;
};

TEST_F(TestEkfUpdate, TestUpdateState) {

  float expected_pos = 0.0f;
  const float expected_vel = 5.0f;
  
  const auto& initial_state = ekf_->State();
  EXPECT_NEAR(initial_state(0), expected_pos, 1e-6);
  EXPECT_NEAR(initial_state(1), expected_vel, 1e-6);

  const float dt = 1.0f;
  for (int i = 0; i < num_iterations_; ++i) {
    expected_pos += expected_vel * dt;

    ekf_->Predict(dt);

    const float m = GenerateMeasurement(expected_pos);
    const Eigen::Matrix<float,1,1> measurement = Eigen::Matrix<float,1,1>::Constant(m);
    ekf_->Update(measurement);

    const auto& state = ekf_->State();

    float sigma = 6 * std::sqrt(measurement_noise_);
    EXPECT_NEAR(state(0), expected_pos, sigma);
    EXPECT_NEAR(state(1), expected_vel, sigma);
  }
}

TEST_F(TestEkfUpdate, TestUpdateCov) {

  float expected_pos = 0.0f;
  const float expected_vel = 5.0f;
  
  const auto& initial_state = ekf_->State();
  EXPECT_NEAR(initial_state(0), expected_pos, 1e-6);
  EXPECT_NEAR(initial_state(1), expected_vel, 1e-6);

  const float dt = 1.0f;
  for (int i = 0; i < num_iterations_; ++i) {
    expected_pos += expected_vel * dt;

    ekf_->Predict(dt);
    Eigen::Array<float,2,2> cov_before = ekf_->Uncertainty().array();

    const float m = GenerateMeasurement(expected_pos);
    const Eigen::Matrix<float,1,1> measurement = Eigen::Matrix<float,1,1>::Constant(m);
    ekf_->Update(measurement);

    Eigen::Array<float,2,2> cov_after = ekf_->Uncertainty().array();

    EXPECT_TRUE((cov_after <= cov_before).all());
  }
}