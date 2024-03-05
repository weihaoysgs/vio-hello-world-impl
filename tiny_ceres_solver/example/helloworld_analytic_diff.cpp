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
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using analytic jacobian matrix.
/// This file is part of VIO-Hello-World
/// Copyright (C) 2023 ZJU
/// You should have received a copy of the GNU General Public License,
/// see <https://www.gnu.org/licenses/> for more details
/// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include <iostream>

#include "tceres/autodiff_cost_function.h"
#include "tceres/autodiff_local_parameterization.h"
#include "tceres/casts.h"
#include "tceres/collections_port.h"
#include "tceres/context.h"
#include "tceres/cost_function.h"
#include "tceres/covariance.h"
#include "tceres/crs_matrix.h"
#include "tceres/evaluation_callback.h"
#include "tceres/execution_summary.h"
#include "tceres/fpclassify.h"
#include "tceres/integral_types.h"
#include "tceres/internal/autodiff.h"
#include "tceres/internal/disable_warnings.h"
#include "tceres/internal/eigen.h"
#include "tceres/internal/fixed_array.h"
#include "tceres/internal/macros.h"
#include "tceres/internal/manual_constructor.h"
#include "tceres/internal/port.h"
#include "tceres/internal/scoped_ptr.h"
#include "tceres/jet.h"
#include "tceres/local_parameterization.h"
#include "tceres/loss_function.h"
#include "tceres/map_util.h"
#include "tceres/mutex.h"
#include "tceres/parameter_block.h"
#include "tceres/problem_impl.h"
#include "tceres/residual_block.h"
#include "tceres/rotation.h"
#include "tceres/sized_cost_function.h"
#include "tceres/sparse_matrix.h"
#include "tceres/stl_util.h"
#include "tceres/stringprintf.h"
#include "tceres/types.h"
#include "tceres/wall_time.h"
#include "tceres/block_structure.h"
#include "tceres/block_sparse_matrix.h"
// A CostFunction implementing analytically derivatives for the
// function f(x) = 10 - x.
class QuadraticCostFunction : public tceres::SizedCostFunction<1 /* number of residuals */, 1 /* size of first parameter */>
{
 public:
  virtual ~QuadraticCostFunction() {}

  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    double x = parameters[0][0];

    // f(x) = 10 - x.
    residuals[0] = 10 - x;

    // f'(x) = -1. Since there's only 1 parameter and that parameter
    // has 1 dimension, there is only 1 element to fill in the
    // jacobians.
    //
    // Since the Evaluate function can be called with the jacobians
    // pointer equal to NULL, the Evaluate function must check to see
    // if jacobians need to be computed.
    //
    // For this simple problem it is overkill to check if jacobians[0]
    // is NULL, but in general when writing more complex
    // CostFunctions, it is possible that Ceres may only demand the
    // derivatives w.r.t. a subset of the parameter blocks.
    if (jacobians != NULL && jacobians[0] != NULL)
    {
      jacobians[0][0] = -1;
    }

    return true;
  }
};
int main(int argc, char** argv)
{
  using namespace tceres::internal;
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;
  // Build the problem.
  tceres::Problem problem;

  // Set up the only cost function (also known as residual).
  tceres::CostFunction* cost_function = new QuadraticCostFunction;
  problem.AddResidualBlock(cost_function, NULL, &x);

  return 0;
}