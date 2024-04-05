option(CATKIN_ENABLE_TESTING OFF)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        sensor_msgs
        cv_bridge
        image_transport
)

include_directories(${catkin_INCLUDE_DIRS})

# if you want to using your opencv, delete the OpenCV_LIBS of ros(noetic,melodic)
# attention please, the cv_bridge will using OpenCV of ros default, if not
# you need to change the cmake file of cv_bridge in your system
# file(GLOB_RECURSE OLD_OPENCV "/usr/lib/x86_64-linux-gnu/libopencv_*")
# list(REMOVE_ITEM catkin_LIBRARIES ${OLD_OPENCV})

set(SOURCE_FILES
        src/map_point.cpp
        src/world_manager.cpp
        src/setting.cpp
        src/camera_calibration.cpp
        src/frame.cpp
        src/visual_frontend.cpp
        src/map_manager.cpp
        src/feat/feat.cpp
        src/feat/good_feature_impl.cpp
        src/feat/orb_cv_impl.cpp
        src/feat/orb_slam_impl.cpp
        src/viz/visualization_base.cpp
        src/viz/rviz_visualization.cpp
        src/viz/camera_visualizer.cpp
        src/viz/pangolin_visualization.cpp
        src/tracker/tracker_base.cpp
        src/tracker/lightglue_impl.cpp
        src/tracker/optical_flow_impl.cpp
        src/mapping.cpp
        src/loop_closer.cpp
        src/optimization.cpp
)

add_library(vio_hw ${SOURCE_FILES})

target_link_libraries(vio_hw
        feature_extraction
        geometry
        tiny_ceres
        feature_tracking
        DBoW2
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
)

add_executable(hello_world_vio node/hello_world_vio.cpp)
add_backward(hello_world_vio)
target_link_libraries(hello_world_vio
        vio_hw
)
add_executable(run_kitti_node node/run_kitti_node.cpp)
add_backward(run_kitti_node)
target_link_libraries(run_kitti_node
        vio_hw
)
add_executable(run_hilti_node node/run_hilti_node.cpp)
add_backward(run_hilti_node)
target_link_libraries(run_hilti_node
        vio_hw
)