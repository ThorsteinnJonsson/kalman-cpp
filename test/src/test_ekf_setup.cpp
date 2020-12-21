#include <gtest/gtest.h>
#include "extended_kalman_filter.h"

using namespace KalmanCpp;




class TestEkfSetup : public ::testing::Test {
 protected:
  static constexpr int StateDim = 2;
  static constexpr int MeasDim = 3;

  struct TestPredictor : public KalmanCpp::BasePredictor<TestPredictor, float, StateDim, KalmanCpp::JacobianMethod::Numerical> {
    template <typename InMat, typename OutMat>
    void GetPrediction(const InMat& in, OutMat& out) const {
      out = OutMat::Zero();
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

 public:
  void SetUp() override {
    ekf_ = std::make_unique<EkfType>();
  }

  void TearDown() override {}  
  
};

TEST_F(TestEkfSetup, CheckStateZeroInit) {
  Eigen::Vector2f init_state = Eigen::Vector2f::Zero();
  Eigen::Matrix2f init_cov = Eigen::Matrix2f::Zero();
  ekf_->InitState(init_state, init_cov);

  auto& state = ekf_->State();
  EXPECT_TRUE(state.isZero(1e-9));
  
  auto& cov = ekf_->Uncertainty();
  EXPECT_TRUE(cov.isZero(1e-9));
}


TEST_F(TestEkfSetup, CheckStateInit) {
  Eigen::Vector2f init_state = {1.1f, 2.2f};
  Eigen::Matrix2f init_cov = Eigen::Matrix2f::Zero();
  ekf_->InitState(init_state, init_cov);

  auto& state = ekf_->State();
  EXPECT_NEAR(state(0), 1.1f, 1e-9);
  EXPECT_NEAR(state(1), 2.2f, 1e-9);
}


TEST_F(TestEkfSetup, CheckCovInit) {
  Eigen::Vector2f init_state = Eigen::Vector2f::Zero();
  Eigen::Matrix2f init_cov;
  init_cov << 1.1f, 2.2f, 3.3f, 4.4f;
  ekf_->InitState(init_state, init_cov);

  auto& cov = ekf_->Uncertainty();
  EXPECT_NEAR(cov(0,0), 1.1f, 1e-9);
  EXPECT_NEAR(cov(0,1), 2.2f, 1e-9);
  EXPECT_NEAR(cov(1,0), 3.3f, 1e-9);
  EXPECT_NEAR(cov(1,1), 4.4f, 1e-9);
}

TEST_F(TestEkfSetup, CheckNoiseZeroInit) {
  Eigen::Matrix2f process_noise = Eigen::Matrix2f::Zero();
  Eigen::Matrix3f measurement_noise = Eigen::Matrix3f::Zero();
  ekf_->InitUncertainty(process_noise, measurement_noise);

  auto& q = ekf_->ProcessNoise();
  EXPECT_TRUE(q.isZero(1e-9));

  auto& r = ekf_->MeasurementNoise();
  EXPECT_TRUE(r.isZero(1e-9));
}

TEST_F(TestEkfSetup, CheckProcessNoiseInit) {
  Eigen::Matrix2f process_noise;
  process_noise << 1.1f, 2.2f, 3.3f, 4.4f;
  Eigen::Matrix3f measurement_noise = Eigen::Matrix3f::Zero();
  ekf_->InitUncertainty(process_noise, measurement_noise);

  auto& q = ekf_->ProcessNoise();
  EXPECT_NEAR(q(0,0), 1.1f, 1e-9);
  EXPECT_NEAR(q(0,1), 2.2f, 1e-9);
  EXPECT_NEAR(q(1,0), 3.3f, 1e-9);
  EXPECT_NEAR(q(1,1), 4.4f, 1e-9);
}


TEST_F(TestEkfSetup, CheckMeasNoiseInit) {
  Eigen::Matrix2f process_noise = Eigen::Matrix2f::Zero();
  Eigen::Matrix3f measurement_noise;
  measurement_noise << 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f;
  ekf_->InitUncertainty(process_noise, measurement_noise);

  auto& r = ekf_->MeasurementNoise();
  EXPECT_NEAR(r(0,0), 1.1f, 1e-9);
  EXPECT_NEAR(r(0,1), 2.2f, 1e-9);
  EXPECT_NEAR(r(0,2), 3.3f, 1e-9);
  EXPECT_NEAR(r(1,0), 4.4f, 1e-9);
  EXPECT_NEAR(r(1,1), 5.5f, 1e-9);
  EXPECT_NEAR(r(1,2), 6.6f, 1e-9);
  EXPECT_NEAR(r(2,0), 7.7f, 1e-9);
  EXPECT_NEAR(r(2,1), 8.8f, 1e-9);
  EXPECT_NEAR(r(2,2), 9.9f, 1e-9);
}

