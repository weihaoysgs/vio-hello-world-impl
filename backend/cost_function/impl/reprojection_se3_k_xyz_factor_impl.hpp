#pragma once

namespace backend {
namespace DirectLeftSE3 {
inline bool ReprojectionErrorKSE3XYZ::Evaluate( double const* const* parameters, double* residuals,
                                                double** jacobians ) const {
  // [fx, fy, cx, cy]
  Eigen::Map<const Eigen::Vector4d> lcalib( parameters[0] );

  // [tx, ty, tz, qw, qx, qy, qz]
  Eigen::Map<const Eigen::Vector3d> twc( parameters[1] );
  Eigen::Map<const Eigen::Quaterniond> qwc( parameters[1] + 3 );

  Sophus::SE3d Twc( qwc, twc );
  Sophus::SE3d Tcw = Twc.inverse();

  // [x,y,z]
  Eigen::Map<const Eigen::Vector3d> wpt( parameters[2] );

  // Compute left/right reproj err
  Eigen::Vector2d pred;

  Eigen::Vector3d lcampt = Tcw * wpt;

  const double linvz = 1. / lcampt.z();

  pred << lcalib( 0 ) * lcampt.x() * linvz + lcalib( 2 ),
      lcalib( 1 ) * lcampt.y() * linvz + lcalib( 3 );

  Eigen::Map<Eigen::Vector2d> werr( residuals );
  werr = sqrt_info_ * ( pred - unpx_ );

  // Update chi2err and depthpos info for
  // post optim checking
  chi2err_ = 0.;
  for ( int i = 0; i < 2; i++ ) chi2err_ += std::pow( residuals[i], 2 );

  // std::cout << "\n chi2 err : " << chi2err_;

  isdepthpositive_ = true;
  if ( lcampt.z() <= 0 ) isdepthpositive_ = false;

  if ( jacobians != NULL ) {
    const double linvz2 = linvz * linvz;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
    J_lcam << linvz * lcalib( 0 ), 0., -lcampt.x() * linvz2 * lcalib( 0 ), 0., linvz * lcalib( 1 ),
        -lcampt.y() * linvz2 * lcalib( 1 );

    Eigen::Matrix<double, 2, 3> J_lRcw;

    if ( jacobians[1] != NULL || jacobians[2] != NULL ) {
      J_lRcw.noalias() = J_lcam * Tcw.rotationMatrix();
    }

    if ( jacobians[0] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_lcalib( jacobians[0] );
      J_lcalib.setZero();
      J_lcalib( 0, 0 ) = lcampt.x() * linvz;
      J_lcalib( 0, 2 ) = 1.;
      J_lcalib( 1, 1 ) = lcampt.y() * linvz;
      J_lcalib( 1, 3 ) = 1.;

      J_lcalib = sqrt_info_ * J_lcalib.eval();
    }
    if ( jacobians[1] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose( jacobians[1] );
      J_se3pose.setZero();

      J_se3pose.block<2, 3>( 0, 0 ) = -1. * J_lRcw;
      J_se3pose.block<2, 3>( 0, 3 ).noalias() = J_lRcw * Sophus::SO3d::hat( wpt );

      J_se3pose = sqrt_info_ * J_se3pose.eval();
    }
    if ( jacobians[2] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_wpt( jacobians[2] );
      J_wpt.setZero();
      J_wpt.block<2, 3>( 0, 0 ) = J_lRcw;

      J_wpt = sqrt_info_ * J_wpt.eval();
    }
  }

  return true;
}
inline bool ReprojectionErrorRightCamKSE3XYZ::Evaluate( double const* const* parameters,
                                                        double* residuals,
                                                        double** jacobians ) const {
  // [fx, fy, cx, cy]
  Eigen::Map<const Eigen::Vector4d> rcalib( parameters[0] );

  // [tx, ty, tz, qw, qx, qy, qz]
  Eigen::Map<const Eigen::Vector3d> twc( parameters[1] );
  Eigen::Map<const Eigen::Quaterniond> qwc( parameters[1] + 3 );

  Eigen::Map<const Eigen::Vector3d> trl( parameters[2] );
  Eigen::Map<const Eigen::Quaterniond> qrl( parameters[2] + 3 );

  Sophus::SE3d Twc( qwc, twc );
  Sophus::SE3d Tcw = Twc.inverse();
  Sophus::SE3d Trl( qrl, trl );

  // [x,y,z]
  Eigen::Map<const Eigen::Vector3d> wpt( parameters[3] );

  // Compute left/right reproj err
  Eigen::Vector2d pred;

  Eigen::Vector3d rcampt = Trl * Tcw * wpt;

  const double rinvz = 1. / rcampt.z();

  pred << rcalib( 0 ) * rcampt.x() * rinvz + rcalib( 2 ),
      rcalib( 1 ) * rcampt.y() * rinvz + rcalib( 3 );

  Eigen::Map<Eigen::Vector2d> werr( residuals );
  werr = sqrt_info_ * ( pred - unpx_ );

  // Update chi2err and depthpos info for
  // post optim checking
  chi2err_ = 0.;
  for ( int i = 0; i < 2; i++ ) chi2err_ += std::pow( residuals[i], 2 );

  isdepthpositive_ = true;
  if ( rcampt.z() <= 0 ) isdepthpositive_ = false;

  if ( jacobians != NULL ) {
    const double rinvz2 = rinvz * rinvz;

    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_rcam;
    J_rcam << rinvz * rcalib( 0 ), 0., -rcampt.x() * rinvz2 * rcalib( 0 ), 0., rinvz * rcalib( 1 ),
        -rcampt.y() * rinvz2 * rcalib( 1 );

    Eigen::Matrix<double, 2, 3> J_rRcw;

    if ( jacobians[1] != NULL || jacobians[3] != NULL ) {
      Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
      J_rRcw.noalias() = J_rcam * Trl.rotationMatrix() * Rcw;
    }

    if ( jacobians[0] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J_rcalib( jacobians[0] );
      J_rcalib.setZero();
      J_rcalib( 0, 0 ) = rcampt.x() * rinvz;
      J_rcalib( 0, 2 ) = 1.;
      J_rcalib( 1, 1 ) = rcampt.y() * rinvz;
      J_rcalib( 1, 3 ) = 1.;

      J_rcalib = sqrt_info_ * J_rcalib.eval();
    }
    if ( jacobians[1] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3pose( jacobians[1] );
      J_se3pose.setZero();

      Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat( wpt );

      J_se3pose.block<2, 3>( 0, 0 ) = -1. * J_rRcw;
      J_se3pose.block<2, 3>( 0, 3 ).noalias() = J_rRcw * skew_wpt;

      J_se3pose = sqrt_info_ * J_se3pose.eval();
    }
    if ( jacobians[2] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > J_se3extrin( jacobians[2] );
      J_se3extrin.setZero();

      // TODO
    }
    if ( jacobians[3] != NULL ) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J_wpt( jacobians[3] );
      J_wpt.setZero();
      J_wpt.block<2, 3>( 0, 0 ) = J_rRcw;

      J_wpt = sqrt_info_ * J_wpt.eval();
    }
  }

  return true;
}

}  // namespace DirectLeftSE3

}  // namespace backend
