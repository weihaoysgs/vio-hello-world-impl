cmake_minimum_required(VERSION 3.10)
project(vio-hello-world)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -O3 -fopenmp -pthread")

add_subdirectory(tiny_ceres_solver)