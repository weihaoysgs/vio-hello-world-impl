add_executable(gauss_newton_method example/gauss_newton_method.cpp)
target_link_libraries(gauss_newton_method tiny_ceres)

add_executable(hello_backend example/hello_backend.cpp ${BACKWARD_ENABLE})
add_backward(hello_backend)
target_link_libraries(hello_backend tiny_ceres ${OpenCV_LIBS})

add_executable(pgo_test example/pgo_test.cpp ${BACKWARD_ENABLE})
add_backward(pgo_test)
target_link_libraries(pgo_test tiny_ceres ${OpenCV_LIBS})