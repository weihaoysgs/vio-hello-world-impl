include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
)

add_library(tiny_ceres
        internal/tceres/stringprintf.cc
        internal/tceres/corrector.cc
        internal/tceres/array_utils.cc
        internal/tceres/residual_block.cc
        internal/tceres/residual_block_utils.cc
        internal/tceres/loss_function.cc
        internal/tceres/local_parameterization.cc
        internal/tceres/linear_operator.cc
        internal/tceres/triplet_sparse_matrix.cc
        internal/tceres/sparse_matrix.cc
        internal/tceres/program.cc
        internal/tceres/compressed_row_sparse_matrix.cc
)

target_link_libraries(tiny_ceres ${THIRD_PARTY_LIBS})

set(TEST_SOURCES small_blas_test array_utils_test loss_function_test
        residual_block_utils_test residual_block_test householder_vector_test
        local_parameterization_test triplet_sparse_matrix_test)

foreach (test_src ${TEST_SOURCES})
    add_executable(${test_src} internal/tceres/${test_src}.cc)
    target_link_libraries(${test_src} ${THIRD_PARTY_LIBS} tiny_ceres)
    add_test(${test_src} ${test_src})
endforeach (test_src)

add_executable(helloworld_analytic_diff example/helloworld_analytic_diff.cpp)
target_link_libraries(helloworld_analytic_diff tiny_ceres)