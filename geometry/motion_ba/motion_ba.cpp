#include "geometry/motion_ba/motion_ba.hpp"

namespace geometry {

bool tceresMotionOnlyBA(
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const std::vector<int> &vscales, Sophus::SE3d &Twc, const int nmaxiter, const float chi2th,
    const bool buse_robust, const bool bapply_l2_after_robust, const Eigen::Matrix3d &K,
    std::vector<int> &voutliersidx ) {
  float fx = K( 0, 0 );
  float fy = K( 1, 1 );
  float cx = K( 0, 2 );
  float cy = K( 1, 2 );
  return tceresMotionOnlyBA( vunkps, vwpts, vscales, Twc, nmaxiter, chi2th, buse_robust,
                             bapply_l2_after_robust, fx, fy, cx, cy, voutliersidx );
}

bool tceresMotionOnlyBA(
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const std::vector<int> &vscales, Sophus::SE3d &Twc, const int nmaxiter, const float chi2th,
    const bool buse_robust, const bool bapply_l2_after_robust, const float fx, const float fy,
    const float cx, const float cy, std::vector<int> &voutliersidx ) {
  assert( vunkps.size() == vwpts.size() );

  tceres::Problem problem;

  double chi2thrtsq = std::sqrt( chi2th );

  tceres::LossFunctionWrapper *loss_function;
  loss_function = new tceres::LossFunctionWrapper( new tceres::HuberLoss( chi2thrtsq ),
                                                   tceres::TAKE_OWNERSHIP );

  if ( !buse_robust ) {
    loss_function->Reset( NULL, tceres::TAKE_OWNERSHIP );
  }

  size_t nbkps = vunkps.size();

  tceres::LocalParameterization *local_parameterization = new backend::SE3LeftParameterization();

  backend::PoseParametersBlock posepar = backend::PoseParametersBlock( 0, Twc );

  problem.AddParameterBlock( posepar.values(), 7, local_parameterization );

  std::vector<backend::DirectLeftSE3::ReprojectionErrorSE3 *> verrors_;
  std::vector<tceres::ResidualBlockId> vrids_;

  for ( size_t i = 0; i < nbkps; i++ ) {
    auto *f = new backend::DirectLeftSE3::ReprojectionErrorSE3(
        vunkps[i].x(), vunkps[i].y(), fx, fy, cx, cy, vwpts.at( i ), std::pow( 2., vscales[i] ) );
    tceres::ResidualBlockId rid = problem.AddResidualBlock( f, loss_function, posepar.values() );

    verrors_.push_back( f );
    vrids_.push_back( rid );
  }

  tceres::Solver::Options options;
  options.linear_solver_type = tceres::DENSE_SCHUR;
  options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;

  options.num_threads = 1;
  options.max_num_iterations = nmaxiter;
  options.max_solver_time_in_seconds = 0.005;
  options.function_tolerance = 1.e-3;

  options.minimizer_progress_to_stdout = false;

  tceres::Solver::Summary summary;
  tceres::Solve( options, &problem, &summary );

  size_t bad = 0;

  for ( size_t i = 0; i < nbkps; i++ ) {
    auto err = verrors_.at( i );
    if ( err->chi2err_ > chi2th || !err->isdepthpositive_ ) {
      if ( bapply_l2_after_robust ) {
        auto rid = vrids_.at( i );
        problem.RemoveResidualBlock( rid );
      }
      voutliersidx.push_back( i );
      bad++;
    }
  }

  if ( bad == nbkps ) {
    return false;
  }

  if ( bapply_l2_after_robust && !voutliersidx.empty() ) {
    loss_function->Reset( nullptr, tceres::TAKE_OWNERSHIP );
    tceres::Solve( options, &problem, &summary );
  }

  Twc = posepar.getPose();

  return summary.IsSolutionUsable();
}

bool opencvP3PRansac(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const int nmaxiter, const float errth, const float fx, const float fy, const bool boptimize,
    Sophus::SE3d &Twc, std::vector<int> &voutliersidx ) {
  assert( bvs.size() == vwpts.size() );

  size_t nb3dpts = bvs.size();

  if ( nb3dpts < 4 ) {
    return false;
  }

  voutliersidx.reserve( nb3dpts );

  std::vector<cv::Point2f> cvbvs;
  cvbvs.reserve( nb3dpts );

  std::vector<cv::Point3f> cvwpts;
  cvwpts.reserve( nb3dpts );

  for ( size_t i = 0; i < nb3dpts; i++ ) {
    cvbvs.push_back(
        cv::Point2f( bvs.at( i ).x() / bvs.at( i ).z(), bvs.at( i ).y() / bvs.at( i ).z() ) );

    cvwpts.push_back( cv::Point3f( vwpts.at( i ).x(), vwpts.at( i ).y(), vwpts.at( i ).z() ) );
  }

  // Using homoegeneous pts here so no dist or calib.
  cv::Mat D;
  cv::Mat K = cv::Mat::eye( 3, 3, CV_32F );

  cv::Mat tvec, rvec;
  cv::Mat inliers;

  bool use_extrinsic_guess = false;
  float confidence = 0.99;

  float focal = ( fx + fy ) / 2.;

  try {
    cv::solvePnPRansac( cvwpts, cvbvs, K, D, rvec, tvec, use_extrinsic_guess, nmaxiter,
                        errth / focal, confidence, inliers, cv::SOLVEPNP_ITERATIVE );
  }
  catch ( cv::Exception &e ) {
    LOG( WARNING ) << "Catching a cv exception: " << e.msg;
    return false;
  }

  if ( inliers.rows == 0 ) {
    return false;
  }

  int k = 0;
  for ( size_t i = 0; i < nb3dpts; i++ ) {
    if ( inliers.at<int>( k ) == (int)i ) {
      k++;
      if ( k == inliers.rows ) {
        k = 0;
      }
    } else {
      voutliersidx.push_back( i );
    }
  }

  if ( voutliersidx.size() >= nb3dpts - 5 ) {
    return false;
  }

  if ( boptimize ) {
    use_extrinsic_guess = true;

    // Filter outliers
    std::vector<cv::Point2f> in_cvbvs;
    in_cvbvs.reserve( inliers.rows );

    std::vector<cv::Point3f> in_cvwpts;
    in_cvwpts.reserve( inliers.rows );

    for ( int i = 0; i < inliers.rows; i++ ) {
      in_cvbvs.push_back( cvbvs.at( inliers.at<int>( i ) ) );
      in_cvwpts.push_back( cvwpts.at( inliers.at<int>( i ) ) );
    }

    if ( in_cvbvs.size() < 6 ) {
      return false;
    }
    cv::solvePnP( in_cvwpts, in_cvbvs, K, D, rvec, tvec, use_extrinsic_guess,
                  cv::SOLVEPNP_ITERATIVE );
  }

  // Store the resulting pose
  cv::Mat cvR;
  cv::Rodrigues( rvec, cvR );

  Eigen::Vector3d tcw;
  Eigen::Matrix3d Rcw;

  cv::cv2eigen( cvR, Rcw );
  cv::cv2eigen( tvec, tcw );

  Twc.translation() = -1. * Rcw.transpose() * tcw;
  Twc.setRotationMatrix( Rcw.transpose() );

  return true;
}

}  // namespace geometry