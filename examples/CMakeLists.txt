cmake_minimum_required(VERSION 3.15)

#
# Project details
#

project(
  ${CMAKE_PROJECT_NAME}Examples
  LANGUAGES CXX
)

verbose_message("Adding examples under ${CMAKE_PROJECT_NAME}Examples...")

#
# Set the sources for the unit tests and add the executable(s)
#

set(test_sources 
  src/example.cpp
)
add_executable(${PROJECT_NAME} ${test_sources})

#
# Set the compiler standard
#

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_PROJECT_NAME}
)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
target_link_libraries_system(${PROJECT_NAME} PRIVATE matplot)
