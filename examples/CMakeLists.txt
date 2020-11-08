cmake_minimum_required(VERSION 3.15)

#
# Project details
#

project(
  ${CMAKE_PROJECT_NAME}Examples
  LANGUAGES CXX
)

verbose_message("Adding examples under ${CMAKE_PROJECT_NAME}Examples...")

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)

#
# Linear Kalman filter example
#
add_executable(KF_EXAMPLE src/example_kf.cpp)
target_compile_features(KF_EXAMPLE PUBLIC cxx_std_17)
target_link_libraries(
  KF_EXAMPLE
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)
target_link_libraries_system(KF_EXAMPLE PRIVATE matplot)

#
# Extended Kalman filter example
#
add_executable(EKF_EXAMPLE src/example_ekf.cpp)
target_compile_features(EKF_EXAMPLE PUBLIC cxx_std_17)
target_link_libraries(
  EKF_EXAMPLE
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)
target_link_libraries_system(EKF_EXAMPLE PRIVATE matplot)


#
# Linear Kalman filter vs Extended Kalman filter example
#
add_executable(KF_VS_EKF_EXAMPLE src/example_kf_vs_ekf.cpp)
target_compile_features(KF_VS_EKF_EXAMPLE PUBLIC cxx_std_17)
target_link_libraries(
  KF_VS_EKF_EXAMPLE
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)
target_link_libraries_system(KF_VS_EKF_EXAMPLE PRIVATE matplot)
