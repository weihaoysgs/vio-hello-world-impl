#ifndef VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_H
#define VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "tceres/local_parameterization.h"
#include "tceres/slam/rotation_utils.h"

namespace tceres {
namespace slam {
class PoseLocalParameterization : public tceres::LocalParameterization
{
  virtual bool Plus(const double *x, const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const;
  virtual int LocalSize() const;
};
}  // namespace slam
}  // namespace tceres

#include "tceres/slam/impl/pose_local_parameterization.hpp"

#endif  // VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_H
