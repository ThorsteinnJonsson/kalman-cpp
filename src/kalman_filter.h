#ifndef KFCPP_KALMAN_FILTER
#define KFCPP_KALMAN_FILTER

namespace KalmanCpp {

class KalmanFilter {
 public:
  KalmanFilter() = default;
  ~KalmanFilter() = default;

  void SayHello() const;
};

}  // namespace KalmanCpp

#endif  // KFCPP_KALMAN_FILTER