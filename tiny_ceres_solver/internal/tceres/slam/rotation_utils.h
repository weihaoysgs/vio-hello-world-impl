#ifndef VIO_HELLO_WORLD_ROTATION_UTILS_H
#define VIO_HELLO_WORLD_ROTATION_UTILS_H

namespace tceres {
namespace slam {

template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> deltaQ(
    const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;

  Eigen::Quaternion<Scalar_t> dq;
  Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
  half_theta /= static_cast<Scalar_t>(2.0);
  dq.w() = static_cast<Scalar_t>(1.0);
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(
    const Eigen::MatrixBase<Derived> &q) {
  Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
  ans << typename Derived::Scalar(0), -q(2), q(1), q(2),
      typename Derived::Scalar(0), -q(0), -q(1), q(0),
      typename Derived::Scalar(0);
  return ans;
}

}  // namespace slam
}  // namespace tceres
#endif  // VIO_HELLO_WORLD_ROTATION_UTILS_H
