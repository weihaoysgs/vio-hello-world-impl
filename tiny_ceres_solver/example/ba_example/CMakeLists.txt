add_executable(ba_demo ba_demo.cpp  parametersse3.cpp)
target_link_libraries(ba_demo tiny_ceres )

add_executable(ba_6d ba_demo_6d.cpp  parametersse3.cpp)
target_link_libraries(ba_6d tiny_ceres)
