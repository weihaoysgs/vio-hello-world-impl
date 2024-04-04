#include "geometry/epipolar/epipolar_constraint.hpp"
namespace geometry {

bool Opencv5ptEssentialMatrix(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2,
    const int nmaxiter, const float errth, const bool boptimize, const float fx, const float fy,
    Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, std::vector<int> &voutliersidx ) {
  assert( bvs1.size() == bvs2.size() );

  size_t nbpts = bvs1.size();

  if ( nbpts < 5 ) {
    return false;
  }

  voutliersidx.reserve( nbpts );

  std::vector<cv::Point2f> cvbvs1, cvbvs2;
  cvbvs1.reserve( nbpts );
  cvbvs2.reserve( nbpts );

  for ( size_t i = 0; i < nbpts; i++ ) {
    cvbvs1.push_back(
        cv::Point2f( bvs1.at( i ).x() / bvs1.at( i ).z(), bvs1.at( i ).y() / bvs1.at( i ).z() ) );

    cvbvs2.push_back(
        cv::Point2f( bvs2.at( i ).x() / bvs2.at( i ).z(), bvs2.at( i ).y() / bvs2.at( i ).z() ) );
  }

  // Using homoegeneous pts here so no dist or calib.
  cv::Mat K = cv::Mat::eye( 3, 3, CV_32F );

  cv::Mat inliers;

  float confidence = 0.99;
  if ( boptimize ) {
    confidence = 0.999;
  }

  float focal = ( fx + fy ) / 2.;

  cv::Mat E =
      cv::findEssentialMat( cvbvs1, cvbvs2, K, cv::RANSAC, confidence, errth / focal, inliers );

  for ( size_t i = 0; i < nbpts; i++ ) {
    if ( !inliers.at<uchar>( i ) ) {
      voutliersidx.push_back( i );
    }
  }

  if ( voutliersidx.size() >= nbpts - 10 ) {
    return false;
  }

  cv::Mat tvec, cvR;

  cv::recoverPose( E, cvbvs1, cvbvs2, K, cvR, tvec, inliers );

  // Store the resulting pose
  Eigen::Vector3d tcw;
  Eigen::Matrix3d Rcw;

  cv::cv2eigen( cvR, Rcw );
  cv::cv2eigen( tvec, tcw );

  twc = -1. * Rcw.transpose() * tcw;
  Rwc = Rcw.transpose();

  return true;
}

}  // namespace geometry
