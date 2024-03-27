find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        cv_bridge
        image_transport
)

include_directories(${catkin_INCLUDE_DIRS})

set(CATKIN_ENABLE_TESTING OFF)

#option(VIS_WITH_RVIZ "Enable visualization with RViz" ON)
#option(VIS_WITH_PANGOLIN "Enable visualization with Pangolin" OFF)
#if(VIS_WITH_RVIZ)
#    add_definitions(-DVIS_WITH_RVIZ)
#endif()
#if(VIS_WITH_PANGOLIN)
#    add_definitions(-DVIS_WITH_PANGOLIN)
#endif()

set(SOURCE_FILES
        src/map_point.cpp
        src/world_manager.cpp
        src/setting.cpp
        src/camera_calibration.cpp
        src/frame.cpp
        src/visual_frontend.cpp
        src/map_manager.cpp
        src/feat/feat.cpp
        src/viz/visualization_base.cpp
        src/viz/rviz_visualization.cpp
        src/viz/camera_visualizer.cpp
        src/viz/pangolin_visualization.cpp
)

add_library(vio_hw ${SOURCE_FILES})

target_link_libraries(vio_hw
        ${THIRD_PARTY_LIBS}
        feature_extraction
        tiny_ceres
        feature_tracking
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
)

add_executable(hello_world_vio node/hello_world_vio.cpp)
add_backward(hello_world_vio)
target_link_libraries(hello_world_vio
        vio_hw
)