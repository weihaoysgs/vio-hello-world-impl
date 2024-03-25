#pragma once

#include <Eigen/Core>

#include "sophus/se3.hpp"

namespace backend {
class PoseParametersBlock
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PoseParametersBlock() = default;

  PoseParametersBlock(const int id, const Sophus::SE3d &T) {
    id_ = id;
    Eigen::Map<Eigen::Vector3d> t(values_);
    Eigen::Map<Eigen::Quaterniond> q(values_ + 3);
    t = T.translation();
    q = T.unit_quaternion();
  }

  PoseParametersBlock(const PoseParametersBlock &block) {
    id_ = block.id_;
    for (size_t i = 0; i < ndim_; i++) {
      values_[i] = block.values_[i];
    }
  }

  PoseParametersBlock &operator=(const PoseParametersBlock &block) {
    id_ = block.id_;
    for (size_t i = 0; i < ndim_; i++) {
      values_[i] = block.values_[i];
    }
    return *this;
  }

  Sophus::SE3d getPose() {
    Eigen::Map<Eigen::Vector3d> t(values_);
    Eigen::Map<Eigen::Quaterniond> q(values_ + 3);
    return Sophus::SE3d(q, t);
  }

  inline double *values() { return values_; }

  static const size_t ndim_ = 7;
  double values_[ndim_] = {0., 0., 0., 0., 0., 0., 0.};
  int id_ = 0.;
};
}  // namespace backend