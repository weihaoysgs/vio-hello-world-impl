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
#include "tceres/collections_port.h"
#include "tceres/context.h"
#include "tceres/cost_function.h"
#include "tceres/covariance.h"
#include "tceres/crs_matrix.h"
#include "tceres/evaluation_callback.h"
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
#include "tceres/parameter_block.h"
#include "tceres/residual_block.h"
#include "tceres/rotation.h"
#include "tceres/sized_cost_function.h"
#include "tceres/stringprintf.h"
#include "tceres/types.h"

class CostOne : public tceres::CostFunction
{
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const { return true; }
};

int main(int argc, char** argv)
{
  using namespace tceres::internal;
  using namespace tceres;
  std::vector<ParameterBlock*> parameter_blocks{};
  CostFunction* cost_function = new CostOne;
  LossFunction* loss_function = new HuberLoss(1.0);
  auto* residual_block = new ResidualBlock(cost_function, loss_function, parameter_blocks, 1);
  return 0;
}