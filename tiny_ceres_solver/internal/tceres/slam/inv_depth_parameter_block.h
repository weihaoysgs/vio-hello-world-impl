#pragma once

#include "Eigen/Core"

namespace tceres {
namespace slam {

class InvDepthParametersBlock
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InvDepthParametersBlock() {}

  explicit InvDepthParametersBlock(const double depth, const int id = -1) : id_(id) {
    values_[0] = 1. / depth;
  }

  double getInvDepth() const { return values_[0]; }

  inline double* values() { return values_; }

 public:
  static const size_t ndim_ = 1;

 private:
  double values_[ndim_] = {0.};
  int id_ = -1;
};

}  // namespace slam
}  // namespace tceres
