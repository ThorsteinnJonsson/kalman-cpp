#include <gtest/gtest.h>

#include "kalman_filter.h"

TEST(TmpAddTest, CheckValues)
{
  KalmanCpp::KalmanFilter kf;
  kf.SayHello();
  EXPECT_TRUE(true);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
