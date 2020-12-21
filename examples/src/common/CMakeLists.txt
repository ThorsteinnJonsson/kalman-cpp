
set(ExampleCommonLib ${CMAKE_PROJECT_NAME}ExamplesCommon)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)


add_library(${ExampleCommonLib}
    example_plotting.cpp
    ball_simulator.cpp
    straight_line_measurements.cpp)

target_compile_features(${ExampleCommonLib} PUBLIC cxx_std_17)
target_link_libraries(
  ${ExampleCommonLib}
  PRIVATE
    ${CMAKE_PROJECT_NAME}
    CONAN_PKG::eigen
)

target_include_directories(${ExampleCommonLib}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)


target_link_libraries_system(${ExampleCommonLib} PUBLIC matplot)
