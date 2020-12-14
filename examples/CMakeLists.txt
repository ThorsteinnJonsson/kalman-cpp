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
# Simple Extended Kalman filter using Jacobians derived numerically
#
add_executable(EXAMPLE_EKF_NUMERICAL src/example_ekf_numerical.cpp)
target_compile_features(EXAMPLE_EKF_NUMERICAL PUBLIC cxx_std_17)
target_link_libraries(
  EXAMPLE_EKF_NUMERICAL
  PRIVATE
    ${CMAKE_PROJECT_NAME}
    CONAN_PKG::eigen
)
target_link_libraries_system(EXAMPLE_EKF_NUMERICAL PRIVATE matplot)

#
# Simple Extended Kalman filter using Jacobians derived analytically
#
add_executable(EXAMPLE_EKF_ANALYTICAL src/example_ekf_analytical.cpp)
target_compile_features(EXAMPLE_EKF_ANALYTICAL PUBLIC cxx_std_17)
target_link_libraries(
  EXAMPLE_EKF_ANALYTICAL
  PRIVATE
    ${CMAKE_PROJECT_NAME}
    CONAN_PKG::eigen
)
target_link_libraries_system(EXAMPLE_EKF_ANALYTICAL PRIVATE matplot)


#
# Extended Kalman filter example 2
#
# add_executable(EKF_EXAMPLE_2 src/example_ekf_2.cpp)
# target_compile_features(EKF_EXAMPLE_2 PUBLIC cxx_std_17)
# target_link_libraries(
#   EKF_EXAMPLE_2
#   PRIVATE
#     ${CMAKE_PROJECT_NAME}
# )
# target_link_libraries_system(EKF_EXAMPLE_2 PRIVATE matplot)
