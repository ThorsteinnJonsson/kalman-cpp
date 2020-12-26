#include <gtest/gtest.h>
#include "extended_kalman_filter.h"


using namespace KalmanCpp;

// State is [x_pos, x_vel]
// Measurement is [x_pos]
static constexpr int StateDim = 2;
static constexpr int MeasDim = 1;

using StateVec = Eigen::Matrix<float,StateDim,1>;
using StateMat = Eigen::Matrix<float,StateDim,StateDim>;

using MeasVec = Eigen::Matrix<float,MeasDim,1>;
using UpdaterJacobian = Eigen::Matrix<float,MeasDim,StateDim>;

struct TestPredictor : public KalmanCpp::BasePredictor<TestPredictor, float, StateDim, KalmanCpp::JacobianMethod::Numerical> {
  template <typename InMat, typename OutMat>
  void GetPrediction(const InMat& in, OutMat& out) const {
    out = OutMat::Zero();
    out(0) = in(0) + in(1) * this->Timestep();
    out(1) = in(1);
  }
};


TEST(TestNumericalJacobian, TestPredictorJacobian) {

  StateMat expected_jacobian = StateMat::Identity();
  expected_jacobian(0,1) = 1.0f;

  TestPredictor predictor;
  StateVec state = StateVec::Zero();
  const float dt = 1.0f;

  auto [x_new, jacobian] = predictor.Predict<StateVec,StateVec,StateMat>(state, dt);

  EXPECT_TRUE(x_new.isZero(1e-9));
  EXPECT_TRUE((jacobian - expected_jacobian).isZero(1e-9));
}


struct TestUpdater : public KalmanCpp::BaseUpdater<TestUpdater, float, StateDim, MeasDim, KalmanCpp::JacobianMethod::Numerical> {
  template <typename InMat, typename OutMat>
  void GetMeasurement(const InMat& in, OutMat& out) const {
    out = OutMat::Zero();
    out(0) = in(0);
  }
};

TEST(TestNumericalJacobian, TestUpdaterJacobian) {

  UpdaterJacobian expected_jacobian;
  expected_jacobian << 1.0f, 0.0f;

  TestUpdater updater;
  StateVec state = StateVec::Zero();

  auto [z, jacobian] = updater.Update<StateVec,MeasVec,UpdaterJacobian>(state);

  EXPECT_TRUE(z.isZero(1e-9));
  EXPECT_TRUE((jacobian - expected_jacobian).isZero(1e-9));
}
