#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

#include "sophus/se3.hpp"
#include "tceres/local_parameterization.h"
#include "tceres/sized_cost_function.h"

namespace backend {
class LeftSE3RelativePoseError : public tceres::SizedCostFunction<6, 7, 7>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LeftSE3RelativePoseError( const Sophus::SE3d& Tc0c1, const double sigma = 1. ) : Tc0c1_( Tc0c1 ) {
    sqrt_cov_ = sigma * Eigen::Matrix<double, 6, 6>::Identity();
    sqrt_info_ = sqrt_cov_.inverse();
  }

  LeftSE3RelativePoseError( const Sophus::SE3d& Tc0c1, const Eigen::Matrix<double, 6, 6> &info ) : Tc0c1_( Tc0c1 ) {
    sqrt_cov_ = Eigen::Matrix<double, 6, 6>::Identity();
    sqrt_info_ = info;
  }

  virtual bool Evaluate( double const* const* parameters, double* residuals,
                         double** jacobians ) const;

  // Mutable var. that will be updated in const Evaluate()
  mutable double chi2err_;
  mutable bool isdepthpositive_;
  Eigen::Matrix<double, 6, 6> sqrt_cov_, sqrt_info_;

private:
  Sophus::SE3d Tc0c1_;
};
}  // namespace backend

#include "backend/cost_function/impl/se3_left_pgo_factor_impl.hpp"