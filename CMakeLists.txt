cmake_minimum_required(VERSION 2.8)
project(cpp)

find_package(Eigen3 REQUIRED)

add_executable(main main.cpp)

include_directories(${EIGEN3_INCLUDE_DIR})

add_library(async_kalman SHARED async_kalman.cpp async_kalman.hpp)

target_link_libraries (main async_kalman)
