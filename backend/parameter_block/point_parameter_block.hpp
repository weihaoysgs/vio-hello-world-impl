#pragma once

#include <Eigen/Core>

namespace backend {
class PointXYZParametersBlock
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointXYZParametersBlock() = default;

  PointXYZParametersBlock(const int id, const Eigen::Vector3d &X) {
    id_ = id;
    Eigen::Map<Eigen::Vector3d>(values_, 3, 1) = X;
  }

  PointXYZParametersBlock(const PointXYZParametersBlock &block) {
    id_ = block.id_;
    for (size_t i = 0; i < ndim_; i++) {
      values_[i] = block.values_[i];
    }
  }

  PointXYZParametersBlock &operator=(const PointXYZParametersBlock &block) {
    id_ = block.id_;
    for (size_t i = 0; i < ndim_; i++) {
      values_[i] = block.values_[i];
    }
    return *this;
  }

  Eigen::Vector3d getPoint() {
    Eigen::Map<Eigen::Vector3d> X(values_);
    return X;
  }

  inline double *values() { return values_; }

  static const size_t ndim_ = 3;
  double values_[ndim_] = {0., 0., 0.};
  int id_ = -1;
};
}  // namespace backend