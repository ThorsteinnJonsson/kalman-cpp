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
add_executable(KF_EXAMPLE_1 src/example_kf_1.cpp)
target_compile_features(KF_EXAMPLE_1 PUBLIC cxx_std_17)
target_link_libraries(
  KF_EXAMPLE_1
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)
target_link_libraries_system(KF_EXAMPLE_1 PRIVATE matplot)

#
# Extended Kalman filter example
#
add_executable(EKF_EXAMPLE_1 src/example_ekf_1.cpp)
target_compile_features(EKF_EXAMPLE_1 PUBLIC cxx_std_17)
target_link_libraries(
  EKF_EXAMPLE_1
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)
target_link_libraries_system(EKF_EXAMPLE_1 PRIVATE matplot)


#
# Linear Kalman filter vs Extended Kalman filter example
#
add_executable(EKF_EXAMPLE_2 src/example_ekf_2.cpp)
target_compile_features(EKF_EXAMPLE_2 PUBLIC cxx_std_17)
target_link_libraries(
  EKF_EXAMPLE_2
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)
target_link_libraries_system(EKF_EXAMPLE_2 PRIVATE matplot)
