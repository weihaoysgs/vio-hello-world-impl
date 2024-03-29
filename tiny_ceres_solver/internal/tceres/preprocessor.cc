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
// Author: sameragarwal@google.com (Sameer Agarwal)

#include "tceres/callbacks.h"
// #include "tceres/gradient_checking_cost_function.h"
// #include "tceres/line_search_preprocessor.h"
#include "tceres/preprocessor.h"
#include "tceres/problem_impl.h"
#include "tceres/solver.h"
#include "tceres/trust_region_preprocessor.h"

namespace tceres {
namespace internal {

Preprocessor* Preprocessor::Create(MinimizerType minimizer_type) {
  if (minimizer_type == TRUST_REGION) {
    return new TrustRegionPreprocessor;
  }

  // TODO changed by weihao
  // if (minimizer_type == LINE_SEARCH) {
  //   return new LineSearchPreprocessor;
  // }

  LOG(FATAL) << "Unknown minimizer_type: " << minimizer_type;
  return NULL;
}

Preprocessor::~Preprocessor() {
}

void ChangeNumThreadsIfNeeded(Solver::Options* options) {
  if (options->num_linear_solver_threads != -1 &&
      options->num_threads != options->num_linear_solver_threads) {
    LOG(WARNING) << "Solver::Options::num_threads = "
                 << options->num_threads
                 << " and Solver::Options::num_linear_solver_threads = "
                 << options->num_linear_solver_threads
                 << ". Solver::Options::num_linear_solver_threads is "
                 << "deprecated and is ignored."
                 << "Solver::Options::num_threads now controls threading "
                 << "behaviour in all of Ceres Solver. "
                 << "This field will go away in Ceres Solver 1.15.0.";
  }

#ifdef CERES_NO_THREADS
  if (options->num_threads > 1) {
    LOG(WARNING)
        << "Neither OpenMP nor TBB support is compiled into this binary; "
        << "only options.num_threads = 1 is supported. Switching "
        << "to single threaded mode.";
    options->num_threads = 1;
  }
#endif  // CERES_NO_THREADS
}

void SetupCommonMinimizerOptions(PreprocessedProblem* pp) {
  const Solver::Options& options = pp->options;
  Program* program = pp->reduced_program.get();

  // Assuming that the parameter blocks in the program have been
  // reordered as needed, extract them into a contiguous vector.
  pp->reduced_parameters.resize(program->NumParameters());
  double* reduced_parameters = pp->reduced_parameters.data();
  program->ParameterBlocksToStateVector(reduced_parameters);

  Minimizer::Options& minimizer_options = pp->minimizer_options;
  minimizer_options = Minimizer::Options(options);
  minimizer_options.evaluator = pp->evaluator;

  if (options.logging_type != SILENT) {
    pp->logging_callback.reset(
        new LoggingCallback(options.minimizer_type,
                            options.minimizer_progress_to_stdout));
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       pp->logging_callback.get());
  }

  if (options.update_state_every_iteration) {
    pp->state_updating_callback.reset(
      new StateUpdatingCallback(program, reduced_parameters));
    // This must get pushed to the front of the callbacks so that it
    // is run before any of the user callbacks.
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       pp->state_updating_callback.get());
  }
}

}  // namespace internal
}  // namespace ceres
