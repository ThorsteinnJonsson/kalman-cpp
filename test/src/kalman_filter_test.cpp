#include <gtest/gtest.h>

#include "kalman_filter.h"

TEST(TmpAddTest, TestToDo)
{
  const size_t state_dim = 2;
  KalmanCpp::KalmanFilter kf(state_dim);
  (void)kf;
  EXPECT_TRUE(true);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
