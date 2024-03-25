cmake_minimum_required(VERSION 3.10)
project(vio-hello-world)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -fopenmp -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(${PROJECT_SOURCE_DIR}/tiny_ceres_solver/include)
include_directories(${PROJECT_SOURCE_DIR}/tiny_ceres_solver/internal)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/backward-cpp)

find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(GFlags REQUIRED)
find_package(OpenCV 4 REQUIRED)
include_directories(
        ${GLOG_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
)

set(THIRD_PARTY_LIBS
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
)

enable_testing()

add_subdirectory(tiny_ceres_solver)
add_subdirectory(feat)
add_subdirectory(backend)
add_subdirectory(geometry)