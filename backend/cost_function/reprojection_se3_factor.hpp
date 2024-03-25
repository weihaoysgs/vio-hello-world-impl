#pragma once
#include "Eigen/Core"
#include "Eigen/Dense"
#include "sophus/se3.hpp"
#include "tceres/sized_cost_function.h"

namespace backend {
namespace DirectLeftSE3 {

class ReprojectionErrorSE3 : public tceres::SizedCostFunction<2, 7>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionErrorSE3(const double u, const double v, double fx, double fy,
                       double cx, double cy, const Eigen::Vector3d& wpt,
                       const double sigma = 1.)
      : unpx_(u, v), wpt_(wpt), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
    sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
    sqrt_info_ = sqrt_cov_.inverse();
  }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const;

  // Mutable var. that will be updated in const Evaluate()
  mutable double chi2err_;
  mutable bool isdepthpositive_;
  Eigen::Matrix2d sqrt_cov_, sqrt_info_;

 private:
  Eigen::Vector2d unpx_;
  Eigen::Vector3d wpt_;
  double fx_, fy_, cx_, cy_;
};
}  // namespace DirectLeftSE3
}  // namespace backend

#include "backend/cost_function/impl/reprojection_se3_factor_impl.hpp"