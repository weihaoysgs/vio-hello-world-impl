#ifndef VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_IMPL_HPP
#define VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_IMPL_HPP
#include "sophus/se3.hpp"
#include "tceres/rotation.h"

namespace tceres {
namespace slam {

// TODO
bool PoseLocalParameterization::Plus(const double* x, const double* delta,
                                     double* x_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = (_q * dq).normalized();

  return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double* x,
                                                double* jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);

  j.topRows<6>().setIdentity();
  j.bottomRows<1>().setZero();

  return true;
}

int PoseLocalParameterization::GlobalSize() const { return 7; }
int PoseLocalParameterization::LocalSize() const { return 6; }
}  // namespace slam
}  // namespace tceres

#endif  // VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_IMPL_HPP
