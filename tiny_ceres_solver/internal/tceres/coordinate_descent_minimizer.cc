// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "tceres/coordinate_descent_minimizer.h"

#if defined(CERES_USE_TBB) || defined(CERES_USE_CXX11_THREADS)
#include "ceres/parallel_for.h"
#endif

#include <iterator>
#include <numeric>
#include <vector>
#include "tceres/evaluator.h"
#include "tceres/linear_solver.h"
#include "tceres/minimizer.h"
#include "tceres/parameter_block.h"
#include "tceres/parameter_block_ordering.h"
#include "tceres/problem_impl.h"
#include "tceres/program.h"
#include "tceres/residual_block.h"
#include "tceres/scoped_thread_token.h"
#include "tceres/solver.h"
#include "tceres/thread_token_provider.h"
#include "tceres/trust_region_minimizer.h"
#include "tceres/trust_region_strategy.h"

namespace tceres {
namespace internal {

using std::map;
using std::max;
using std::min;
using std::set;
using std::string;
using std::vector;

CoordinateDescentMinimizer::CoordinateDescentMinimizer(ContextImpl* context)
    : context_(CHECK_NOTNULL(context)) {}

CoordinateDescentMinimizer::~CoordinateDescentMinimizer() {
}

bool CoordinateDescentMinimizer::Init(
    const Program& program,
    const ProblemImpl::ParameterMap& parameter_map,
    const ParameterBlockOrdering& ordering,
    string* error) {
  parameter_blocks_.clear();
  independent_set_offsets_.clear();
  independent_set_offsets_.push_back(0);

  // Serialize the OrderedGroups into a vector of parameter block
  // offsets for parallel access.
  map<ParameterBlock*, int> parameter_block_index;
  map<int, set<double*> > group_to_elements = ordering.group_to_elements();
  for (map<int, set<double*> >::const_iterator it = group_to_elements.begin();
       it != group_to_elements.end();
       ++it) {
    for (set<double*>::const_iterator ptr_it = it->second.begin();
         ptr_it != it->second.end();
         ++ptr_it) {
      parameter_blocks_.push_back(parameter_map.find(*ptr_it)->second);
      parameter_block_index[parameter_blocks_.back()] =
          parameter_blocks_.size() - 1;
    }
    independent_set_offsets_.push_back(
        independent_set_offsets_.back() + it->second.size());
  }

  // The ordering does not have to contain all parameter blocks, so
  // assign zero offsets/empty independent sets to these parameter
  // blocks.
  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    if (!ordering.IsMember(parameter_blocks[i]->mutable_user_state())) {
      parameter_blocks_.push_back(parameter_blocks[i]);
      independent_set_offsets_.push_back(independent_set_offsets_.back());
    }
  }

  // Compute the set of residual blocks that depend on each parameter
  // block.
  residual_blocks_.resize(parameter_block_index.size());
  const vector<ResidualBlock*>& residual_blocks = program.residual_blocks();
  for (int i = 0; i < residual_blocks.size(); ++i) {
    ResidualBlock* residual_block = residual_blocks[i];
    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    for (int j = 0; j < num_parameter_blocks; ++j) {
      ParameterBlock* parameter_block = residual_block->parameter_blocks()[j];
      const map<ParameterBlock*, int>::const_iterator it =
          parameter_block_index.find(parameter_block);
      if (it != parameter_block_index.end()) {
        residual_blocks_[it->second].push_back(residual_block);
      }
    }
  }

  evaluator_options_.linear_solver_type = DENSE_QR;
  evaluator_options_.num_eliminate_blocks = 0;
  evaluator_options_.num_threads = 1;
  evaluator_options_.context = context_;

  return true;
}

void CoordinateDescentMinimizer::Minimize(
    const Minimizer::Options& options,
    double* parameters,
    Solver::Summary* summary) {
  // Set the state and mark all parameter blocks constant.
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks_[i];
    parameter_block->SetState(parameters + parameter_block->state_offset());
    parameter_block->SetConstant();
  }

  scoped_array<LinearSolver*> linear_solvers(
      new LinearSolver*[options.num_threads]);

  LinearSolver::Options linear_solver_options;
  linear_solver_options.type = DENSE_QR;
  linear_solver_options.context = context_;

  for (int i = 0; i < options.num_threads; ++i) {
    linear_solvers[i] = LinearSolver::Create(linear_solver_options);
  }

  for (int i = 0; i < independent_set_offsets_.size() - 1; ++i) {
    const int num_problems =
        independent_set_offsets_[i + 1] - independent_set_offsets_[i];
    // Avoid parallelization overhead call if the set is empty.
    if (num_problems == 0) {
      continue;
    }

    const int num_inner_iteration_threads =
        min(options.num_threads, num_problems);
    evaluator_options_.num_threads =
        max(1, options.num_threads / num_inner_iteration_threads);

    ThreadTokenProvider thread_token_provider(num_inner_iteration_threads);

#ifdef CERES_USE_OPENMP
    // The parameter blocks in each independent set can be optimized
    // in parallel, since they do not co-occur in any residual block.
#pragma omp parallel for num_threads(num_inner_iteration_threads)
#endif

#if !(defined(CERES_USE_TBB) || defined(CERES_USE_CXX11_THREADS))
    for (int j = independent_set_offsets_[i];
         j < independent_set_offsets_[i + 1];
         ++j) {
#else
    ParallelFor(context_,
                independent_set_offsets_[i],
                independent_set_offsets_[i + 1],
                num_inner_iteration_threads,
                [&](int j) {
#endif // !(defined(CERES_USE_TBB) || defined(CERES_USE_CXX11_THREADS))

      const ScopedThreadToken scoped_thread_token(&thread_token_provider);
      const int thread_id = scoped_thread_token.token();

      ParameterBlock* parameter_block = parameter_blocks_[j];
      const int old_index = parameter_block->index();
      const int old_delta_offset = parameter_block->delta_offset();
      parameter_block->SetVarying();
      parameter_block->set_index(0);
      parameter_block->set_delta_offset(0);

      Program inner_program;
      inner_program.mutable_parameter_blocks()->push_back(parameter_block);
      *inner_program.mutable_residual_blocks() = residual_blocks_[j];

      // TODO(sameeragarwal): Better error handling. Right now we
      // assume that this is not going to lead to problems of any
      // sort. Basically we should be checking for numerical failure
      // of some sort.
      //
      // On the other hand, if the optimization is a failure, that in
      // some ways is fine, since it won't change the parameters and
      // we are fine.
      Solver::Summary inner_summary;
      Solve(&inner_program,
            linear_solvers[thread_id],
            parameters + parameter_block->state_offset(),
            &inner_summary);

      parameter_block->set_index(old_index);
      parameter_block->set_delta_offset(old_delta_offset);
      parameter_block->SetState(parameters + parameter_block->state_offset());
      parameter_block->SetConstant();
    }
#if defined(CERES_USE_TBB) || defined(CERES_USE_CXX11_THREADS)
  );
#endif
  }

  for (int i =  0; i < parameter_blocks_.size(); ++i) {
    parameter_blocks_[i]->SetVarying();
  }

  for (int i = 0; i < options.num_threads; ++i) {
    delete linear_solvers[i];
  }
}

// Solve the optimization problem for one parameter block.
void CoordinateDescentMinimizer::Solve(Program* program,
                                       LinearSolver* linear_solver,
                                       double* parameter,
                                       Solver::Summary* summary) {
  *summary = Solver::Summary();
  summary->initial_cost = 0.0;
  summary->fixed_cost = 0.0;
  summary->final_cost = 0.0;
  string error;

  Minimizer::Options minimizer_options;
  minimizer_options.evaluator.reset(
      CHECK_NOTNULL(Evaluator::Create(evaluator_options_, program, &error)));
  minimizer_options.jacobian.reset(
      CHECK_NOTNULL(minimizer_options.evaluator->CreateJacobian()));

  TrustRegionStrategy::Options trs_options;
  trs_options.linear_solver = linear_solver;
  minimizer_options.trust_region_strategy.reset(
      CHECK_NOTNULL(TrustRegionStrategy::Create(trs_options)));
  minimizer_options.is_silent = true;

  TrustRegionMinimizer minimizer;
  minimizer.Minimize(minimizer_options, parameter, summary);
}

bool CoordinateDescentMinimizer::IsOrderingValid(
    const Program& program,
    const ParameterBlockOrdering& ordering,
    string* message) {
  const map<int, set<double*> >& group_to_elements =
      ordering.group_to_elements();

  // Verify that each group is an independent set
  map<int, set<double*> >::const_iterator it = group_to_elements.begin();
  for (; it != group_to_elements.end(); ++it) {
    if (!program.IsParameterBlockSetIndependent(it->second)) {
      *message =
          StringPrintf("The user-provided "
                       "parameter_blocks_for_inner_iterations does not "
                       "form an independent set. Group Id: %d", it->first);
      return false;
    }
  }
  return true;
}

// Find a recursive decomposition of the Hessian matrix as a set
// of independent sets of decreasing size and invert it. This
// seems to work better in practice, i.e., Cameras before
// points.
ParameterBlockOrdering* CoordinateDescentMinimizer::CreateOrdering(
    const Program& program) {
  scoped_ptr<ParameterBlockOrdering> ordering(new ParameterBlockOrdering);
  ComputeRecursiveIndependentSetOrdering(program, ordering.get());
  ordering->Reverse();
  return ordering.release();
}

}  // namespace internal
}  // namespace ceres
