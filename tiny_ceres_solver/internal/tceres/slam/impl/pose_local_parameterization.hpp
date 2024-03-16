#ifndef VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_IMPL_HPP
#define VIO_HELLO_WORLD_POSE_LOCAL_PARAMETERIZATION_IMPL_HPP
#include "tceres/rotation.h"

namespace tceres {
namespace slam {
inline bool QuaternionPlus(const double* x, const double* delta,
                           double* x_plus_delta) {
  const double norm_delta =
      sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  if (norm_delta > 0.0) {
    const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
    double q_delta[4];
    q_delta[0] = cos(norm_delta);
    q_delta[1] = sin_delta_by_delta * delta[0];
    q_delta[2] = sin_delta_by_delta * delta[1];
    q_delta[3] = sin_delta_by_delta * delta[2];
    QuaternionProduct(q_delta, x, x_plus_delta);
  } else {
    for (int i = 0; i < 4; ++i) {
      x_plus_delta[i] = x[i];
    }
  }
  return true;
}

template <int N = 3>
bool IdentityPlus(const double* x, const double* delta, double* x_plus_delta) {
  VectorRef(x_plus_delta, N) = ConstVectorRef(x, N) + ConstVectorRef(delta, N);
  return true;
}

// TODO
bool PoseLocalParameterization::Plus(const double* x, const double* delta,
                                     double* x_plus_delta) const {
  int x_cursor = 0;
  int delta_cursor = 0;
  if (!QuaternionPlus(x + x_cursor, delta + delta_cursor,
                      x_plus_delta + x_cursor)) {
    return false;
  }
  delta_cursor += 3;
  x_cursor += 4;
  if (!IdentityPlus(x + x_cursor, delta + delta_cursor,
                    x_plus_delta + x_cursor)) {
    return false;
  }
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
