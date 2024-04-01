add_library(geometry motion_ba/motion_ba.cpp)
target_link_libraries(geometry tiny_ceres)
add_executable(triangulate_test example/triangulate_test.cpp)
target_link_libraries(triangulate_test ${OpenCV_LIBS})