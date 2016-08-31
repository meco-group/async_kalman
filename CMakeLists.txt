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
add_executable(main main.cpp)

include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

add_library(async_kalman SHARED async_kalman.cpp async_kalman.hpp)
target_link_libraries (async_kalman ${SLICOT_LIB})

target_link_libraries (tests async_kalman pthread)
target_link_libraries (demo async_kalman pthread)
target_link_libraries (main pthread)
