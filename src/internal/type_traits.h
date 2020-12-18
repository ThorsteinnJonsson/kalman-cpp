#ifndef KALMAN_CPP_TYPE_TRAITS_H
#define KALMAN_CPP_TYPE_TRAITS_H

#include <type_traits>

namespace KalmanCpp::type_traits {

template <typename T>
struct has_get_prediction {
  struct Dummy {};

  template <typename C, typename Dummy1, typename Dummy2>
  static auto Test(Dummy1* d1, Dummy2* d2)
      -> decltype(std::declval<C>().GetPrediction(*d1, *d2), std::true_type());

  template <typename, typename, typename>
  static std::false_type Test(...);

  typedef decltype(Test<T, Dummy, Dummy>(nullptr, nullptr)) type;
  static const bool value =
      std::is_same<std::true_type, type>::value;
};

template <typename T> 
inline constexpr bool has_get_prediction_v = has_get_prediction<T>::value;



template <typename T>
struct has_apply_control_input {
  struct Dummy {};

  template <typename C, typename Dummy1>
  static auto Test(Dummy1* d1, Dummy1* d2)
      -> decltype(std::declval<C>().GetControlInput(*d1, *d2), std::true_type());

  template <typename, typename>
  static std::false_type Test(...);

  typedef decltype(Test<T, Dummy>(nullptr, nullptr)) type;
  static const bool value =
      std::is_same<std::true_type, type>::value;
};

template <typename T> 
inline constexpr bool has_apply_control_input_v = has_apply_control_input<T>::value;



template <typename T>
struct has_get_measurement {
  struct Dummy {};

  template <typename C, typename Dummy1, typename Dummy2>
  static auto Test(Dummy1* d1, Dummy2* d2)
      -> decltype(std::declval<C>().GetMeasurement(*d1, *d2), std::true_type());

  template <typename, typename, typename>
  static std::false_type Test(...);

  typedef decltype(Test<T, Dummy, Dummy>(nullptr, nullptr)) type;
  static const bool value =
      std::is_same<std::true_type, type>::value;
};

template <typename T> 
inline constexpr bool has_get_measurement_v = has_get_measurement<T>::value;



template <typename T>
struct has_get_jacobian {
  struct Dummy {};

  template <typename C, typename Dummy1, typename Dummy2>
  static auto Test(Dummy1* d1, Dummy2* d2) -> decltype(std::declval<C>().GetJacobian(*d1, *d2), std::true_type());

  template <typename, typename, typename>
  static std::false_type Test(...);

  typedef decltype(Test<T, Dummy, Dummy>(nullptr, nullptr)) type;
  static const bool value =
      std::is_same<std::true_type, type>::value;
};

template <typename T> 
inline constexpr bool has_get_jacobian_v = has_get_jacobian<T>::value;

}  // namespace KalmanCpp

#endif  // KALMAN_CPP_TYPE_TRAITS_H