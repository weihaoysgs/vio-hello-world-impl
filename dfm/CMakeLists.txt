

add_library(tensorrt_utils
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt_utils/tensorrt_utils/logger.cpp
)

add_library(deep_feature_matcher
        src/superpoint.cpp
        src/lightglue.cpp
)

target_link_libraries(deep_feature_matcher
        ${OpenCV_LIBS}
        nvinfer
        nvonnxparser
        ${CUDA_LIBRARIES}
        tensorrt_utils
)

add_executable(superpoint_test example/superpoint_test.cpp)
target_link_libraries(superpoint_test deep_feature_matcher gflags)
add_executable(lightglue_test example/lightglue_test.cpp)
target_link_libraries(lightglue_test deep_feature_matcher gflags)