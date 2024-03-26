find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        cv_bridge
        image_transport
)
include_directories(${catkin_INCLUDE_DIRS})

add_library(vio_hw
        src/map_point.cpp
        src/world_manager.cpp
        src/setting.cpp
        src/camera_calibration.cpp
        src/frame.cpp
        src/visual_frontend.cpp
        src/map_manager.cpp
        internal/feat/feat.cpp
)

target_link_libraries(vio_hw ${THIRD_PARTY_LIBS} feature_extraction)

add_executable(hello_world_vio node/hello_world_vio.cpp)
add_backward(hello_world_vio)
target_link_libraries(hello_world_vio
        tiny_ceres
        feature_extraction
        feature_tracking
        ${THIRD_PARTY_LIBS}
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
        vio_hw
)