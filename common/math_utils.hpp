#ifndef VIO_HELLO_WORLD_MATH_UTILS_HPP
#define VIO_HELLO_WORLD_MATH_UTILS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

namespace com {
inline Eigen::Matrix3d g2R( const Eigen::Vector3d &g ) {
  Eigen::Matrix3d R0;
  Eigen::Vector3d ng1 = g.normalized();
  Eigen::Vector3d ng2{ 0, 0, 1.0 };
  ng2 = ng2.normalized();
  R0 = Eigen::Quaterniond::FromTwoVectors( ng1, ng2 ).toRotationMatrix();
  double yaw = R0.eulerAngles( 2, 1, 0 )( 0 );
  R0 = Eigen::AngleAxisd( -yaw, Eigen::Vector3d::UnitZ() ) * R0;
  return R0;
}

inline Eigen::Vector3d R2ypr( const Eigen::Matrix3d &R ) {
  Eigen::Vector3d n = R.col( 0 );
  Eigen::Vector3d o = R.col( 1 );
  Eigen::Vector3d a = R.col( 2 );

  Eigen::Vector3d ypr( 3 );
  double y = atan2( n( 1 ), n( 0 ) );
  double p = atan2( -n( 2 ), n( 0 ) * cos( y ) + n( 1 ) * sin( y ) );
  double r = atan2( a( 0 ) * sin( y ) - a( 1 ) * cos( y ), -o( 0 ) * sin( y ) + o( 1 ) * cos( y ) );
  ypr( 0 ) = y;
  ypr( 1 ) = p;
  ypr( 2 ) = r;

  return ypr / M_PI * 180.0;
}
}  // namespace com
#endif  // VIO_HELLO_WORLD_MATH_UTILS_HPP
