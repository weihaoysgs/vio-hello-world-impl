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
// Author: keir@google.com (Keir Mierle)

#include <vector>
#include "tceres/block_evaluate_preparer.h"
#include "tceres/block_jacobian_writer.h"
#include "tceres/compressed_row_jacobian_writer.h"
#include "tceres/compressed_row_sparse_matrix.h"
#include "tceres/crs_matrix.h"
#include "tceres/dense_jacobian_writer.h"
#include "tceres/dynamic_compressed_row_finalizer.h"
#include "tceres/dynamic_compressed_row_jacobian_writer.h"
#include "tceres/evaluator.h"
#include "tceres/internal/port.h"
#include "tceres/program_evaluator.h"
#include "tceres/scratch_evaluate_preparer.h"
#include "glog/logging.h"

namespace tceres {
namespace internal {

Evaluator::~Evaluator() {}

Evaluator* Evaluator::Create(const Evaluator::Options& options,
                             Program* program,
                             std::string* error) {
  CHECK(options.context != NULL);

  switch (options.linear_solver_type) {
    case DENSE_QR:
    case DENSE_NORMAL_CHOLESKY:
      return new ProgramEvaluator<ScratchEvaluatePreparer,
                                  DenseJacobianWriter>(options,
                                                       program);
    case DENSE_SCHUR:
    case SPARSE_SCHUR:
    case ITERATIVE_SCHUR:
    case CGNR:
      return new ProgramEvaluator<BlockEvaluatePreparer,
                                  BlockJacobianWriter>(options,
                                                       program);
    case SPARSE_NORMAL_CHOLESKY:
      if (options.dynamic_sparsity) {
        return new ProgramEvaluator<ScratchEvaluatePreparer,
                                    DynamicCompressedRowJacobianWriter,
                                    DynamicCompressedRowJacobianFinalizer>(
                                        options, program);
      } else {
        return new ProgramEvaluator<BlockEvaluatePreparer,
                                    BlockJacobianWriter>(options,
                                                         program);
      }

    default:
      *error = "Invalid Linear Solver Type. Unable to create evaluator.";
      return NULL;
  }
}

}  // namespace internal
}  // namespace ceres
