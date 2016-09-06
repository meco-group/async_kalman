cmake_minimum_required(VERSION 2.8)
project(cpp)

find_library(SLICOT_LIB
  slicot
  PATHS $ENV{SLICOT})

if(NOT SLICOT_LIB)
  message(STATUS "SLICOT: Cound not find library slicot. Try stetting SLICOT_LIBRARY_DIR env var.")
endif()

find_package(Eigen3 REQUIRED)

add_executable(tests tests.cpp)
add_executable(demo demo.cpp)
add_executable(simple_test simple_test.cpp)

include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -Wall")

add_library(async_kalman SHARED kalman_filter.hpp kalman_operations.cpp kalman_operations.hpp kalman_odometry.cpp kalman_odometry_impl.hpp kalman_odometry.hpp kalman_kinematic.hpp)
target_link_libraries (async_kalman ${SLICOT_LIB})

target_link_libraries (tests async_kalman)
target_link_libraries (demo async_kalman)
target_link_libraries (simple_test async_kalman)
