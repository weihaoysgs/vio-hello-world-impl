/// A simple example of using the Ceres minimizer.
/// This file is part of VIO-Hello-World
/// Copyright (C) 2023 ZJU
/// You should have received a copy of the GNU General Public License,
/// see <https://www.gnu.org/licenses/> for more details
/// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include "iostream"
#include "random"
#include "tceres/problem.h"
#include "tceres/sized_cost_function.h"
#include "tceres/solver.h"
#include "tceres/stringprintf.h"
#include "tceres/parameter_block.h"

class CurveFittingCostFunction : public tceres::SizedCostFunction<1, 3>
{
 public:
  CurveFittingCostFunction(double x, double y) : obs_x_(x), obs_y_(y) {}
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    double ae = parameters[0][0];
    double be = parameters[0][1];
    double ce = parameters[0][2];

    double estimate = std::exp(ae * obs_x_ * obs_x_ + be * obs_x_ + ce);
    residuals[0] = estimate - obs_y_;

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = estimate * obs_x_ * obs_x_;
      jacobians[0][1] = estimate * obs_x_;
      jacobians[0][2] = estimate;
    }
    return true;
  }

 private:
  const double obs_x_, obs_y_;
};

double uniform_rand(double lowerBndr, double upperBndr) {
  return lowerBndr +
         ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}
double gauss_rand(double mean, double sigma) {
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging("curve_fitting_analytic");
  FLAGS_stderrthreshold = google::INFO;

  double ar = 1.0, br = 2.0, cr = 1.0;   // real value
  double ae = 2.0, be = -1.0, ce = 5.0;  // estimate init value
  int N = 100;                           // sample number
  double w_sigma = 0.5;                  // noise of sigma
  auto curve_function = [&](double x) -> double {
    // exp(ax^2 + bx + c)
    return std::exp(ar * x * x + br * x + cr) +
           gauss_rand(0.0, w_sigma * w_sigma);
  };

  std::vector<std::pair<double, double>> sample_data;
  for (int i = 0; i < N; ++i) {
    double x = i / 100.0;
    sample_data.emplace_back(x, curve_function(x));
  }

  double abc[3] = {ae, be, ce};
  tceres::Problem problem;
  // this step is not necessary
  problem.AddParameterBlock(abc, 3);

  for (int i = 0; i < N; ++i) {
    tceres::CostFunction* cost_function = new CurveFittingCostFunction(
        sample_data[i].first, sample_data[i].second);
    problem.AddResidualBlock(cost_function, nullptr, abc);
  }

  tceres::Solver::Summary summary;
  tceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = tceres::DOGLEG;
  options.linear_solver_type = tceres::DENSE_SCHUR;
  options.minimizer_type = tceres::TRUST_REGION;
  tceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport();
  std::cout << tceres::internal::StringPrintf("\nreal abc: %.3f %.3f %.3f", ar,
                                              br, cr);
  std::cout << tceres::internal::StringPrintf("\nestimate abc: %.3f %.3f %.3f",
                                              abc[0], abc[1], abc[2]);
}