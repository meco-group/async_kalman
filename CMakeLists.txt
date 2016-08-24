cmake_minimum_required(VERSION 2.8)
project(cpp)
add_executable(main main.cpp)

add_library(async_kalman SHARED async_kalman.cpp async_kalman.hpp)

target_link_libraries (main async_kalman)
