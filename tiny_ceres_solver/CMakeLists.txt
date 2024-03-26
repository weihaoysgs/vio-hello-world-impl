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
        internal/tceres/wall_time.cc
        internal/tceres/context.cc
        internal/tceres/context_impl.cc
        internal/tceres/problem.cc
        internal/tceres/problem_impl.cc
        internal/tceres/scratch_evaluate_preparer.cc
        internal/tceres/block_structure.cc
        internal/tceres/block_sparse_matrix.cc
        internal/tceres/block_evaluate_preparer.cc
        internal/tceres/dense_sparse_matrix.cc
        internal/tceres/block_jacobian_writer.cc
        internal/tceres/dynamic_compressed_row_jacobian_writer.cc
        internal/tceres/compressed_row_jacobian_writer.cc
        internal/tceres/thread_token_provider.cc
        internal/tceres/evaluator.cc
        internal/tceres/dynamic_compressed_row_sparse_matrix.cc
        internal/tceres/evaluator_test_utils.cc
        internal/tceres/solver.cc
        internal/tceres/types.cc
        internal/tceres/parameter_block_ordering.cc
        internal/tceres/trust_region_step_evaluator.cc
        internal/tceres/block_random_access_matrix.cc
        internal/tceres/block_random_access_diagonal_matrix.cc
        internal/tceres/schur_eliminator.cc
        internal/tceres/block_random_access_dense_matrix.cc
        internal/tceres/block_random_access_sparse_matrix.cc
        internal/tceres/conjugate_gradients_solver.cc
        internal/tceres/detect_structure.cc
        internal/tceres/lapack.cc
        internal/tceres/file.cc
        internal/tceres/linear_least_squares_problems.cc
        internal/tceres/test_util.cc
        internal/tceres/polynomial.cc
        internal/tceres/function_sample.cc
        internal/tceres/dogleg_strategy.cc
        internal/tceres/callbacks.cc
        internal/tceres/schur_templates.cc
        internal/tceres/is_close.cc
        internal/tceres/gradient_checker.cc
# Important
        internal/tceres/linear_solver.cc
        internal/tceres/schur_complement_solver.cc
        internal/tceres/eigensparse.cc
        internal/tceres/sparse_cholesky.cc
        internal/tceres/minimizer.cc
        internal/tceres/trust_region_strategy.cc
        internal/tceres/line_search.cc
        internal/tceres/coordinate_descent_minimizer.cc
        internal/tceres/trust_region_minimizer.cc
        internal/tceres/preprocessor.cc
        internal/tceres/reorder_program.cc
        internal/tceres/trust_region_preprocessor.cc
        internal/tceres/gradient_checking_cost_function.cc
        internal/tceres/levenberg_marquardt_strategy.cc
)

target_link_libraries(tiny_ceres ${THIRD_PARTY_LIBS})

set(TEST_SOURCES small_blas_test array_utils_test loss_function_test
        residual_block_utils_test residual_block_test householder_vector_test
        local_parameterization_test triplet_sparse_matrix_test
        evaluator_test problem_test parameter_block_ordering_test
        graph_algorithms_test block_random_access_diagonal_matrix_test
        block_random_access_sparse_matrix_test conjugate_gradients_solver_test
        detect_structure_test block_random_access_dense_matrix_test
        invert_psd_matrix_test trust_region_preprocessor_test
        gradient_checking_cost_function_test
)

foreach (test_src ${TEST_SOURCES})
    add_executable(${test_src} internal/tceres/${test_src}.cc)
    target_link_libraries(${test_src} ${THIRD_PARTY_LIBS} tiny_ceres)
    add_test(${test_src} ${test_src})
endforeach (test_src)

set(EXE_EXAMPLE
    helloworld_analytic_diff curve_fitting_analytic ba_problem_analytic
)
foreach (exe_node ${EXE_EXAMPLE})
    add_executable(${exe_node} example/${exe_node}.cc)
    target_link_libraries(${exe_node} tiny_ceres)
endforeach ()

#add_subdirectory(example/ba_example)