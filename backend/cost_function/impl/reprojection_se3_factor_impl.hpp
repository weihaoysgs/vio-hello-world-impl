#pragma once

namespace backend {
namespace DirectLeftSE3 {
bool ReprojectionErrorSE3::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
  // [tx, ty, tz, qw, qx, qy, qz]
  Eigen::Map<const Eigen::Vector3d> twc(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> qwc(parameters[0] + 3);

  Sophus::SE3d Twc(qwc, twc);
  Sophus::SE3d Tcw = Twc.inverse();

  // Compute left/right reproj err
  Eigen::Vector2d pred;

  Eigen::Vector3d lcampt = Tcw * wpt_;

  const double linvz = 1. / lcampt.z();

  pred << fx_ * lcampt.x() * linvz + cx_, fy_ * lcampt.y() * linvz + cy_;

  Eigen::Map<Eigen::Vector2d> werr(residuals);
  werr = sqrt_info_ * (pred - unpx_);

  // Update chi2err and depthpos info for
  // post optim checking
  chi2err_ = 0.;
  for (int i = 0; i < 2; i++) chi2err_ += std::pow(residuals[i], 2);

  isdepthpositive_ = true;
  if (lcampt.z() <= 0) isdepthpositive_ = false;

  if (jacobians != NULL) {
    const double linvz2 = linvz * linvz;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
    J_lcam << linvz * fx_, 0., -lcampt.x() * linvz2 * fx_, 0., linvz * fy_,
        -lcampt.y() * linvz2 * fy_;

    Eigen::Matrix<double, 2, 3> J_lRcw;
    J_lRcw.noalias() = J_lcam * Tcw.rotationMatrix();

    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose(
          jacobians[0]);
      J_se3pose.setZero();

      J_se3pose.block<2, 3>(0, 0).noalias() = -1. * J_lRcw;
      J_se3pose.block<2, 3>(0, 3).noalias() = J_lRcw * Sophus::SO3d::hat(wpt_);

      J_se3pose = sqrt_info_ * J_se3pose.eval();
    }
  }

  return true;
}
}  // namespace DirectLeftSE3

}  // namespace backend
