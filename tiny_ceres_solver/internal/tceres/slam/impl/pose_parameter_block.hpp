#ifndef VIO_HELLO_WORLD_POSE_PARAMETER_BLOCK_IMPL_HPP
#define VIO_HELLO_WORLD_POSE_PARAMETER_BLOCK_IMPL_HPP

namespace tceres {
namespace slam {

PoseParameterBlock::PoseParameterBlock(const Eigen::Quaterniond &Q,
                                       const Eigen::Vector3d &trans, int id) {
  id_ = id;
  Eigen::Map<Eigen::Vector3d> t(values_);
  Eigen::Map<Eigen::Quaterniond> q(values_ + 3);
  t = trans;
  q = Q;
}

double *PoseParameterBlock::values() { return values_; }

Eigen::Quaterniond PoseParameterBlock::getQuaternion() const {
  Eigen::Map<const Eigen::Quaterniond> q(values_ + 3);
  return q;
}

Eigen::Matrix3d PoseParameterBlock::getRotation() const {
  return getQuaternion().toRotationMatrix();
}

Eigen::Vector3d PoseParameterBlock::getTranslation() const {
  Eigen::Map<const Eigen::Vector3d> t(values_);
  return t;
}

}  // namespace slam
}  // namespace tceres

#endif  // VIO_HELLO_WORLD_POSE_PARAMETER_BLOCK_IMPL_HPP
