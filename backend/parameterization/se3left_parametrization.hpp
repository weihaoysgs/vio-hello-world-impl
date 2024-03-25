
#pragma once

#include <Eigen/Core>

#include "sophus/se3.hpp"
#include "tceres/local_parameterization.h"
#include "tceres/sized_cost_function.h"

namespace backend {

class SE3LeftParameterization : public tceres::LocalParameterization
{
 public:
  virtual bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> t(x);
    Eigen::Map<const Eigen::Quaterniond> q(x + 3);

    Eigen::Map<const Eigen::Matrix<double, 6, 1>> vdelta(delta);

    // Left update
    Sophus::SE3d upT = Sophus::SE3d::exp(vdelta) * Sophus::SE3d(q, t);

    Eigen::Map<Eigen::Vector3d> upt(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> upq(x_plus_delta + 3);

    upt = upT.translation();
    upq = upT.unit_quaternion();

    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J.topRows<6>().setIdentity();
    J.bottomRows<1>().setZero();
    return true;
  }

  virtual int GlobalSize() const { return 7; }
  virtual int LocalSize() const { return 6; }
};
}  // namespace backend