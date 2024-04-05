#pragma once
namespace backend {
bool LeftSE3RelativePoseError::Evaluate( double const* const* parameters, double* residuals,
                                         double** jacobians ) const {
  // [tx, ty, tz, qw, qx, qy, qz]
  Eigen::Map<const Eigen::Vector3d> twc0( parameters[0] );
  Eigen::Map<const Eigen::Quaterniond> qwc0( parameters[0] + 3 );

  // [tx, ty, tz, qw, qx, qy, qz]
  Eigen::Map<const Eigen::Vector3d> twc1( parameters[1] );
  Eigen::Map<const Eigen::Quaterniond> qwc1( parameters[1] + 3 );

  Sophus::SE3d Twc0( qwc0, twc0 );
  Sophus::SE3d Twc1( qwc1, twc1 );
  Sophus::SE3d Tc1w = Twc1.inverse();

  Sophus::SE3d Tc1c0 = Tc1w * Twc0;

  Sophus::SE3d err = ( Tc1c0 * Tc0c1_ );

  Eigen::Matrix<double, 6, 1> verr = err.log();

  Eigen::Map<Eigen::Matrix<double, 6, 1>> werr( residuals );
  werr = sqrt_info_ * verr;

  // Update chi2err info for
  // post optim checking
  chi2err_ = 0.;
  for ( int i = 0; i < 6; i++ ) {
    chi2err_ += std::pow( residuals[i], 2 );
  }

  if ( jacobians != nullptr ) {
    Eigen::Matrix3d skew_rho = Sophus::SO3d::hat( verr.block<3, 1>( 0, 0 ) );
    Eigen::Matrix3d skew_omega = Sophus::SO3d::hat( err.so3().log() );

    Eigen::Matrix<double, 6, 6> I6x6 = Eigen::Matrix<double, 6, 6>::Identity();

    if ( jacobians[0] != nullptr ) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_Tc0( jacobians[0] );
      J_Tc0.setZero();

      // Adapted from Strasdat PhD Appendix

      Eigen::Matrix<double, 6, 6> J_c0;
      J_c0.setZero();
      J_c0.block<3, 3>( 0, 0 ).noalias() = -1. * skew_omega;
      J_c0.block<3, 3>( 0, 3 ).noalias() = -1. * skew_rho;
      J_c0.block<3, 3>( 3, 3 ).noalias() = -1. * skew_omega;

      J_Tc0.block<6, 6>( 0, 0 ).noalias() = ( I6x6 + 0.5 * J_c0 ) * Tc1w.Adj();

      J_Tc0 = sqrt_info_ * J_Tc0.eval();
    }
    if ( jacobians[1] != nullptr ) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_Tc1( jacobians[1] );
      J_Tc1.setZero();

      Eigen::Matrix<double, 6, 6> J_c1;
      J_c1.setZero();
      J_c1.block<3, 3>( 0, 0 ).noalias() = skew_omega;
      J_c1.block<3, 3>( 0, 3 ).noalias() = skew_rho;
      J_c1.block<3, 3>( 3, 3 ).noalias() = skew_omega;

      J_Tc1.block<6, 6>( 0, 0 ).noalias() =
          -1. * ( I6x6 + 0.5 * J_c1 ) * ( Twc0 * Tc0c1_ ).inverse().Adj();

      J_Tc1 = sqrt_info_ * J_Tc1.eval();
    }
  }

  return true;
}

}  // namespace backend
