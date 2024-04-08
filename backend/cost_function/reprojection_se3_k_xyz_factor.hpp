#pragma once
#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "tceres/sized_cost_function.h"
namespace backend {
namespace DirectLeftSE3 {

class ReprojectionErrorKSE3XYZ : public tceres::SizedCostFunction<2, 4, 7, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionErrorKSE3XYZ( const double u, const double v, const double sigma = 1. )
      : unpx_( u, v ) {
    sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
    sqrt_info_ = sqrt_cov_.inverse();
  }

  virtual bool Evaluate( double const* const* parameters, double* residuals,
                         double** jacobians ) const;

  // Mutable var. that will be updated in const Evaluate()
  mutable double chi2err_;
  mutable bool isdepthpositive_;
  Eigen::Matrix2d sqrt_cov_, sqrt_info_;

private:
  Eigen::Vector2d unpx_;
};

class ReprojectionErrorRightCamKSE3XYZ : public tceres::SizedCostFunction<2, 4, 7, 7, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionErrorRightCamKSE3XYZ( const double u, const double v, const double sigma = 1. )
      : unpx_( u, v ) {
    sqrt_cov_ = sigma * Eigen::Matrix2d::Identity();
    sqrt_info_ = sqrt_cov_.inverse();
  }

  virtual bool Evaluate( double const* const* parameters, double* residuals,
                         double** jacobians ) const;

  // Mutable var. that will be updated in const Evaluate()
  mutable double chi2err_;
  mutable bool isdepthpositive_;
  Eigen::Matrix2d sqrt_cov_, sqrt_info_;

private:
  Eigen::Vector2d unpx_;
};
}  // namespace DirectLeftSE3
}  // namespace backend

#include "backend/cost_function/impl/reprojection_se3_k_xyz_factor_impl.hpp"