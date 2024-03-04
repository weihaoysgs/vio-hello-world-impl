cmake_minimum_required(VERSION 3.10)
project(vio-hello-world)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++17 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(GFlags REQUIRED)

include_directories(
        ${GLOG_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIRS}
)

set(THIRD_PARTY_LIBS
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
)

enable_testing()

add_subdirectory(tiny_ceres_solver)