#include "tceres/solver.h"

#include <sstream>  // NOLINT
#include <vector>

#include "iostream"
#include "tceres/casts.h"
#include "tceres/context.h"
#include "tceres/context_impl.h"
// #include "tceres/detect_structure.h"
// #include "tceres/gradient_checking_cost_function.h"
#include "tceres/internal/port.h"
#include "tceres/parameter_block_ordering.h"
// #include "tceres/preprocessor.h"
#include "tceres/problem.h"
#include "tceres/problem_impl.h"
#include "tceres/program.h"
// #include "tceres/schur_templates.h"
// #include "tceres/solver_utils.h"
#include "tceres/stringprintf.h"
#include "tceres/types.h"
#include "tceres/wall_time.h"

namespace tceres {
namespace {

using std::map;
using std::string;
using std::vector;

#define OPTION_OP(x, y, OP)                                          \
  if (!(options.x OP y))                                             \
  {                                                                  \
    std::stringstream ss;                                            \
    ss << "Invalid configuration. ";                                 \
    ss << string("Solver::Options::" #x " = ") << options.x << ". "; \
    ss << "Violated constraint: ";                                   \
    ss << string("Solver::Options::" #x " " #OP " " #y);             \
    *error = ss.str();                                               \
    return false;                                                    \
  }

#define OPTION_OP_OPTION(x, y, OP)                                   \
  if (!(options.x OP options.y))                                     \
  {                                                                  \
    std::stringstream ss;                                            \
    ss << "Invalid configuration. ";                                 \
    ss << string("Solver::Options::" #x " = ") << options.x << ". "; \
    ss << string("Solver::Options::" #y " = ") << options.y << ". "; \
    ss << "Violated constraint: ";                                   \
    ss << string("Solver::Options::" #x);                            \
    ss << string(#OP " Solver::Options::" #y ".");                   \
    *error = ss.str();                                               \
    return false;                                                    \
  }

#define OPTION_GE(x, y) OPTION_OP(x, y, >=);
#define OPTION_GT(x, y) OPTION_OP(x, y, >);
#define OPTION_LE(x, y) OPTION_OP(x, y, <=);
#define OPTION_LT(x, y) OPTION_OP(x, y, <);
#define OPTION_LE_OPTION(x, y) OPTION_OP_OPTION(x, y, <=)
#define OPTION_LT_OPTION(x, y) OPTION_OP_OPTION(x, y, <)

bool CommonOptionsAreValid(const Solver::Options& options, string* error)
{
  OPTION_GE(max_num_iterations, 0);
  OPTION_GE(max_solver_time_in_seconds, 0.0);
  OPTION_GE(function_tolerance, 0.0);
  OPTION_GE(gradient_tolerance, 0.0);
  OPTION_GE(parameter_tolerance, 0.0);
  OPTION_GT(num_threads, 0);
  if (options.check_gradients)
  {
    OPTION_GT(gradient_check_relative_precision, 0.0);
    OPTION_GT(gradient_check_numeric_derivative_relative_step_size, 0.0);
  }
  return true;
}

bool TrustRegionOptionsAreValid(const Solver::Options& options, string* error)
{
  OPTION_GT(initial_trust_region_radius, 0.0);
  OPTION_GT(min_trust_region_radius, 0.0);
  OPTION_GT(max_trust_region_radius, 0.0);
  OPTION_LE_OPTION(min_trust_region_radius, max_trust_region_radius);
  OPTION_LE_OPTION(min_trust_region_radius, initial_trust_region_radius);
  OPTION_LE_OPTION(initial_trust_region_radius, max_trust_region_radius);
  OPTION_GE(min_relative_decrease, 0.0);
  OPTION_GE(min_lm_diagonal, 0.0);
  OPTION_GE(max_lm_diagonal, 0.0);
  OPTION_LE_OPTION(min_lm_diagonal, max_lm_diagonal);
  OPTION_GE(max_num_consecutive_invalid_steps, 0);
  OPTION_GT(eta, 0.0);
  OPTION_GE(min_linear_solver_iterations, 0);
  OPTION_GE(max_linear_solver_iterations, 1);
  OPTION_LE_OPTION(min_linear_solver_iterations, max_linear_solver_iterations);

  if (options.use_inner_iterations)
  {
    OPTION_GE(inner_iteration_tolerance, 0.0);
  }

  if (options.use_inner_iterations && options.evaluation_callback != NULL)
  {
    *error =
        "Inner iterations (use_inner_iterations = true) can't be "
        "combined with an evaluation callback "
        "(options.evaluation_callback != NULL).";
    return false;
  }

  if (options.use_nonmonotonic_steps)
  {
    OPTION_GT(max_consecutive_nonmonotonic_steps, 0);
  }

  if (options.linear_solver_type == ITERATIVE_SCHUR && options.use_explicit_schur_complement && options.preconditioner_type != SCHUR_JACOBI)
  {
    *error =
        "use_explicit_schur_complement only supports "
        "SCHUR_JACOBI as the preconditioner.";
    return false;
  }

  if (options.preconditioner_type == CLUSTER_JACOBI && options.sparse_linear_algebra_library_type != SUITE_SPARSE)
  {
    *error =
        "CLUSTER_JACOBI requires "
        "Solver::Options::sparse_linear_algebra_library_type to be "
        "SUITE_SPARSE.";
    return false;
  }

  if (options.preconditioner_type == CLUSTER_TRIDIAGONAL && options.sparse_linear_algebra_library_type != SUITE_SPARSE)
  {
    *error =
        "CLUSTER_TRIDIAGONAL requires "
        "Solver::Options::sparse_linear_algebra_library_type to be "
        "SUITE_SPARSE.";
    return false;
  }

#ifdef CERES_NO_LAPACK
  if (options.dense_linear_algebra_library_type == LAPACK)
  {
    if (options.linear_solver_type == DENSE_NORMAL_CHOLESKY)
    {
      *error =
          "Can't use DENSE_NORMAL_CHOLESKY with LAPACK because "
          "LAPACK was not enabled when Ceres was built.";
      return false;
    }
    else if (options.linear_solver_type == DENSE_QR)
    {
      *error =
          "Can't use DENSE_QR with LAPACK because "
          "LAPACK was not enabled when Ceres was built.";
      return false;
    }
    else if (options.linear_solver_type == DENSE_SCHUR)
    {
      *error =
          "Can't use DENSE_SCHUR with LAPACK because "
          "LAPACK was not enabled when Ceres was built.";
      return false;
    }
  }
#endif

#ifdef CERES_NO_SUITESPARSE
  if (options.sparse_linear_algebra_library_type == SUITE_SPARSE)
  {
    if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY)
    {
      *error =
          "Can't use SPARSE_NORMAL_CHOLESKY with SUITESPARSE because "
          "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }
    else if (options.linear_solver_type == SPARSE_SCHUR)
    {
      *error =
          "Can't use SPARSE_SCHUR with SUITESPARSE because "
          "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }
    else if (options.preconditioner_type == CLUSTER_JACOBI)
    {
      *error =
          "CLUSTER_JACOBI preconditioner not supported. "
          "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }
    else if (options.preconditioner_type == CLUSTER_TRIDIAGONAL)
    {
      *error =
          "CLUSTER_TRIDIAGONAL preconditioner not supported. "
          "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }
  }
#endif

#ifdef CERES_NO_CXSPARSE
  if (options.sparse_linear_algebra_library_type == CX_SPARSE)
  {
    if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY)
    {
      *error =
          "Can't use SPARSE_NORMAL_CHOLESKY with CX_SPARSE because "
          "CXSparse was not enabled when Ceres was built.";
      return false;
    }
    else if (options.linear_solver_type == SPARSE_SCHUR)
    {
      *error =
          "Can't use SPARSE_SCHUR with CX_SPARSE because "
          "CXSparse was not enabled when Ceres was built.";
      return false;
    }
  }
#endif

#ifndef CERES_USE_EIGEN_SPARSE
  if (options.sparse_linear_algebra_library_type == EIGEN_SPARSE)
  {
    if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY)
    {
      *error =
          "Can't use SPARSE_NORMAL_CHOLESKY with EIGEN_SPARSE because "
          "Eigen's sparse linear algebra was not enabled when Ceres was "
          "built.";
      return false;
    }
    else if (options.linear_solver_type == SPARSE_SCHUR)
    {
      *error =
          "Can't use SPARSE_SCHUR with EIGEN_SPARSE because "
          "Eigen's sparse linear algebra was not enabled when Ceres was "
          "built.";
      return false;
    }
  }
#endif

  if (options.sparse_linear_algebra_library_type == NO_SPARSE)
  {
    if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY)
    {
      *error =
          "Can't use SPARSE_NORMAL_CHOLESKY as "
          "sparse_linear_algebra_library_type is NO_SPARSE.";
      return false;
    }
    else if (options.linear_solver_type == SPARSE_SCHUR)
    {
      *error =
          "Can't use SPARSE_SCHUR as "
          "sparse_linear_algebra_library_type is NO_SPARSE.";
      return false;
    }
  }

  if (options.trust_region_strategy_type == DOGLEG)
  {
    if (options.linear_solver_type == ITERATIVE_SCHUR || options.linear_solver_type == CGNR)
    {
      *error =
          "DOGLEG only supports exact factorization based linear "
          "solvers. If you want to use an iterative solver please "
          "use LEVENBERG_MARQUARDT as the trust_region_strategy_type";
      return false;
    }
  }

  if (options.trust_region_minimizer_iterations_to_dump.size() > 0 && options.trust_region_problem_dump_format_type != CONSOLE &&
      options.trust_region_problem_dump_directory.empty())
  {
    *error = "Solver::Options::trust_region_problem_dump_directory is empty.";
    return false;
  }

  if (options.dynamic_sparsity && options.linear_solver_type != SPARSE_NORMAL_CHOLESKY)
  {
    *error = "Dynamic sparsity is only supported with SPARSE_NORMAL_CHOLESKY.";
    return false;
  }

  return true;
}

bool LineSearchOptionsAreValid(const Solver::Options& options, string* error)
{
  OPTION_GT(max_lbfgs_rank, 0);
  OPTION_GT(min_line_search_step_size, 0.0);
  OPTION_GT(max_line_search_step_contraction, 0.0);
  OPTION_LT(max_line_search_step_contraction, 1.0);
  OPTION_LT_OPTION(max_line_search_step_contraction, min_line_search_step_contraction);
  OPTION_LE(min_line_search_step_contraction, 1.0);
  OPTION_GT(max_num_line_search_step_size_iterations, 0);
  OPTION_GT(line_search_sufficient_function_decrease, 0.0);
  OPTION_LT_OPTION(line_search_sufficient_function_decrease, line_search_sufficient_curvature_decrease);
  OPTION_LT(line_search_sufficient_curvature_decrease, 1.0);
  OPTION_GT(max_line_search_step_expansion, 1.0);

  if ((options.line_search_direction_type == tceres::BFGS || options.line_search_direction_type == tceres::LBFGS) &&
      options.line_search_type != tceres::WOLFE)
  {
    *error = string("Invalid configuration: Solver::Options::line_search_type = ") + string(LineSearchTypeToString(options.line_search_type)) +
             string(
                 ". When using (L)BFGS, "
                 "Solver::Options::line_search_type must be set to WOLFE.");
    return false;
  }

  // Warn user if they have requested BISECTION interpolation, but constraints
  // on max/min step size change during line search prevent bisection scaling
  // from occurring. Warn only, as this is likely a user mistake, but one which
  // does not prevent us from continuing.
  LOG_IF(WARNING, (options.line_search_interpolation_type == tceres::BISECTION &&
                   (options.max_line_search_step_contraction > 0.5 || options.min_line_search_step_contraction < 0.5)))
      << "Line search interpolation type is BISECTION, but specified "
      << "max_line_search_step_contraction: " << options.max_line_search_step_contraction << ", and "
      << "min_line_search_step_contraction: " << options.min_line_search_step_contraction
      << ", prevent bisection (0.5) scaling, continuing with solve regardless.";

  return true;
}

#undef OPTION_OP
#undef OPTION_OP_OPTION
#undef OPTION_GT
#undef OPTION_GE
#undef OPTION_LE
#undef OPTION_LT
#undef OPTION_LE_OPTION
#undef OPTION_LT_OPTION

void StringifyOrdering(const vector<int>& ordering, string* report)
{
  if (ordering.size() == 0)
  {
    internal::StringAppendF(report, "AUTOMATIC");
    return;
  }

  for (int i = 0; i < ordering.size() - 1; ++i)
  {
    internal::StringAppendF(report, "%d,", ordering[i]);
  }
  internal::StringAppendF(report, "%d", ordering.back());
}

void SummarizeGivenProgram(const internal::Program& program, Solver::Summary* summary)
{
  summary->num_parameter_blocks = program.NumParameterBlocks();
  summary->num_parameters = program.NumParameters();
  summary->num_effective_parameters = program.NumEffectiveParameters();
  summary->num_residual_blocks = program.NumResidualBlocks();
  summary->num_residuals = program.NumResiduals();
}

void SummarizeReducedProgram(const internal::Program& program, Solver::Summary* summary)
{
  summary->num_parameter_blocks_reduced = program.NumParameterBlocks();
  summary->num_parameters_reduced = program.NumParameters();
  summary->num_effective_parameters_reduced = program.NumEffectiveParameters();
  summary->num_residual_blocks_reduced = program.NumResidualBlocks();
  summary->num_residuals_reduced = program.NumResiduals();
}

void PreSolveSummarize(const Solver::Options& options,
                       const internal::ProblemImpl* problem,
                       Solver::Summary* summary) {
  SummarizeGivenProgram(problem->program(), summary);
  internal::OrderingToGroupSizes(options.linear_solver_ordering.get(),
                                 &(summary->linear_solver_ordering_given));
  internal::OrderingToGroupSizes(options.inner_iteration_ordering.get(),
                                 &(summary->inner_iteration_ordering_given));

  summary->dense_linear_algebra_library_type  = options.dense_linear_algebra_library_type;  //  NOLINT
  summary->dogleg_type                        = options.dogleg_type;
  summary->inner_iteration_time_in_seconds    = 0.0;
  summary->num_line_search_steps              = 0;
  summary->line_search_cost_evaluation_time_in_seconds = 0.0;
  summary->line_search_gradient_evaluation_time_in_seconds = 0.0;
  summary->line_search_polynomial_minimization_time_in_seconds = 0.0;
  summary->line_search_total_time_in_seconds  = 0.0;
  summary->inner_iterations_given             = options.use_inner_iterations;
  summary->line_search_direction_type         = options.line_search_direction_type;         //  NOLINT
  summary->line_search_interpolation_type     = options.line_search_interpolation_type;     //  NOLINT
  summary->line_search_type                   = options.line_search_type;
  summary->linear_solver_type_given           = options.linear_solver_type;
  summary->max_lbfgs_rank                     = options.max_lbfgs_rank;
  summary->minimizer_type                     = options.minimizer_type;
  summary->nonlinear_conjugate_gradient_type  = options.nonlinear_conjugate_gradient_type;  //  NOLINT
  summary->num_linear_solver_threads_given    = options.num_threads;
  summary->num_threads_given                  = options.num_threads;
  summary->preconditioner_type_given          = options.preconditioner_type;
  summary->sparse_linear_algebra_library_type = options.sparse_linear_algebra_library_type; //  NOLINT
  summary->trust_region_strategy_type         = options.trust_region_strategy_type;         //  NOLINT
  summary->visibility_clustering_type         = options.visibility_clustering_type;         //  NOLINT
}

}  // namespace
bool Solver::Options::IsValid(string* error) const
{
  if (!CommonOptionsAreValid(*this, error))
  {
    return false;
  }

  if (minimizer_type == TRUST_REGION && !TrustRegionOptionsAreValid(*this, error))
  {
    return false;
  }

  // We do not know if the problem is bounds constrained or not, if it
  // is then the trust region solver will also use the line search
  // solver to do a projection onto the box constraints, so make sure
  // that the line search options are checked independent of what
  // minimizer algorithm is being used.
  return LineSearchOptionsAreValid(*this, error);
}
Solver::~Solver() {}

void Solver::Solve(const Solver::Options& options, Problem* problem, Solver::Summary* summary)
{
  // using internal::PreprocessedProblem;
  using internal::Preprocessor;
  using internal::ProblemImpl;
  using internal::Program;
  using internal::scoped_ptr;
  using internal::WallTimeInSeconds;
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(summary);
  double start_time = WallTimeInSeconds();
  *summary = Summary();
  if (!options.IsValid(&summary->message))
  {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  ProblemImpl* problem_impl = problem->problem_impl_.get();
  Program* program = problem_impl->mutable_program();
  PreSolveSummarize(options, problem_impl, summary);
}

void Solve(const Solver::Options& options, Problem* problem, Solver::Summary* summary)
{
  LOG(ERROR) << "Hello world";
  Solver solver;
  solver.Solve(options, problem, summary);
}

Solver::Summary::Summary()
    // Invalid values for most fields, to ensure that we are not
    // accidentally reporting default values.
    : minimizer_type(TRUST_REGION),
      termination_type(FAILURE),
      message("ceres::Solve was not called."),
      initial_cost(-1.0),
      final_cost(-1.0),
      fixed_cost(-1.0),
      num_successful_steps(-1),
      num_unsuccessful_steps(-1),
      num_inner_iteration_steps(-1),
      num_line_search_steps(-1),
      preprocessor_time_in_seconds(-1.0),
      minimizer_time_in_seconds(-1.0),
      postprocessor_time_in_seconds(-1.0),
      total_time_in_seconds(-1.0),
      linear_solver_time_in_seconds(-1.0),
      num_linear_solves(-1),
      residual_evaluation_time_in_seconds(-1.0),
      num_residual_evaluations(-1),
      jacobian_evaluation_time_in_seconds(-1.0),
      num_jacobian_evaluations(-1),
      inner_iteration_time_in_seconds(-1.0),
      line_search_cost_evaluation_time_in_seconds(-1.0),
      line_search_gradient_evaluation_time_in_seconds(-1.0),
      line_search_polynomial_minimization_time_in_seconds(-1.0),
      line_search_total_time_in_seconds(-1.0),
      num_parameter_blocks(-1),
      num_parameters(-1),
      num_effective_parameters(-1),
      num_residual_blocks(-1),
      num_residuals(-1),
      num_parameter_blocks_reduced(-1),
      num_parameters_reduced(-1),
      num_effective_parameters_reduced(-1),
      num_residual_blocks_reduced(-1),
      num_residuals_reduced(-1),
      is_constrained(false),
      num_threads_given(-1),
      num_threads_used(-1),
      num_linear_solver_threads_given(-1),
      num_linear_solver_threads_used(-1),
      linear_solver_type_given(SPARSE_NORMAL_CHOLESKY),
      linear_solver_type_used(SPARSE_NORMAL_CHOLESKY),
      inner_iterations_given(false),
      inner_iterations_used(false),
      preconditioner_type_given(IDENTITY),
      preconditioner_type_used(IDENTITY),
      visibility_clustering_type(CANONICAL_VIEWS),
      trust_region_strategy_type(LEVENBERG_MARQUARDT),
      dense_linear_algebra_library_type(EIGEN),
      sparse_linear_algebra_library_type(SUITE_SPARSE),
      line_search_direction_type(LBFGS),
      line_search_type(ARMIJO),
      line_search_interpolation_type(BISECTION),
      nonlinear_conjugate_gradient_type(FLETCHER_REEVES),
      max_lbfgs_rank(-1)
{
}

}  // namespace tceres