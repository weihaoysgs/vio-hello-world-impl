#ifndef VIO_HELLO_WORLD_POSE_PARAMETER_BLOCK_H
#define VIO_HELLO_WORLD_POSE_PARAMETER_BLOCK_H

#include "Eigen/Core"
#include "Eigen/Dense"
namespace tceres {
namespace slam {

class PoseParameterBlock
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PoseParameterBlock(const Eigen::Quaterniond &Q, const Eigen::Vector3d &t,
                     int id = -1);

  double *values();

  Eigen::Vector3d getTranslation() const;

  Eigen::Quaterniond getQuaternion() const;

  Eigen::Matrix3d getRotation() const;

 public:
  static const size_t ndim_ = 7;

 private:
  double values_[ndim_] = {0., 0., 0., 0., 0., 0., 0.};
  int id_ = 0.;
};
}  // namespace slam
}  // namespace tceres
#include "tceres/slam/impl/pose_parameter_block.hpp"

#endif  // VIO_HELLO_WORLD_POSE_PARAMETER_BLOCK_H
