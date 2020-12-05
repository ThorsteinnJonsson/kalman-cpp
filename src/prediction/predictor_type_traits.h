#ifndef KALMAN_CPP_PREDICTOR_TYPE_TRAITS_H
#define KALMAN_CPP_PREDICTOR_TYPE_TRAITS_H

#include <type_traits>

namespace KalmanCpp {

template <typename T>
struct has_GetPrediction {
  struct dummy {};

  template <typename C, typename P, typename R> static auto test(P* p, R* r) -> decltype(std::declval<C>().GetPrediction(*p, *r), std::true_type());
  template <typename, typename, typename> static std::false_type test(...);

  typedef decltype(test<T, dummy, dummy>(nullptr, nullptr)) type;
  static const bool value =  std::is_same<std::true_type, decltype(test<T,dummy, dummy>(nullptr, nullptr))>::value;
};


template <typename T>
struct has_GetJacobian {
  struct dummy {};

  template <typename C, typename P, typename R> static auto test(P* p, R* r) -> decltype(std::declval<C>().GetJacobian(*p, *r), std::true_type());
  template <typename, typename, typename> static std::false_type test(...);

  typedef decltype(test<T, dummy, dummy>(nullptr, nullptr)) type;
  static const bool value =  std::is_same<std::true_type, decltype(test<T,dummy, dummy>(nullptr, nullptr))>::value;
};


} // namespace KalmanCpp


#endif // KALMAN_CPP_PREDICTOR_TYPE_TRAITS_H