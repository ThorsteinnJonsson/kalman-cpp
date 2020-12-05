#ifndef KALMAN_CPP_PREDICTOR_TYPE_TRAITS_H
#define KALMAN_CPP_PREDICTOR_TYPE_TRAITS_H

#include <type_traits>

namespace KalmanCpp {

template <typename T>
struct has_get_prediction {
  struct dummy {};

  template <typename C, typename Dummy1, typename Dummy2>
  static auto Test(Dummy1* d1, Dummy2* d2)
      -> decltype(std::declval<C>().GetPrediction(*d1, *d2), std::true_type());

  template <typename, typename, typename>
  static std::false_type Test(...);

  typedef decltype(Test<T, dummy, dummy>(nullptr, nullptr)) type;
  static const bool value =
      std::is_same<std::true_type, type>::value;
};

template <typename T>
struct has_get_jacobian {
  struct dummy {};

  template <typename C, typename Dummy1, typename Dummy2>
  static auto Test(Dummy1* d1, Dummy2* d2) -> decltype(std::declval<C>().GetJacobian(*d1, *d2), std::true_type());

  template <typename, typename, typename>
  static std::false_type Test(...);

  typedef decltype(Test<T, dummy, dummy>(nullptr, nullptr)) type;
  static const bool value =
      std::is_same<std::true_type, type>::value;
};

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_PREDICTOR_TYPE_TRAITS_H