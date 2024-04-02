cmake_minimum_required(VERSION 2.8...3.22)
project(vio-hello-world)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -fopenmp -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Unless you know what you are doing, please comment out the following statement
add_definitions("-w")

include_directories(
        ${PROJECT_SOURCE_DIR}/tiny_ceres_solver/include
        ${PROJECT_SOURCE_DIR}/tiny_ceres_solver/internal
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
        ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2
        ${PROJECT_SOURCE_DIR}
)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/backward-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/DBoW2)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        gtest
        gtest_main
        glog
        gflags
)

enable_testing()

add_subdirectory(tiny_ceres_solver)
add_subdirectory(feat)
add_subdirectory(backend)
add_subdirectory(geometry)
add_subdirectory(vio_hw)