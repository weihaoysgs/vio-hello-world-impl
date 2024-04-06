#include "vio_hw/internal/visual_frontend.hpp"

namespace viohw {

VisualFrontEnd::VisualFrontEnd( viohw::SettingPtr state, viohw::FramePtr frame,
                                viohw::MapManagerPtr map, viohw::TrackerBasePtr tracker,
                                VisualizationBasePtr viz )
    : param_( state ),
      current_frame_( frame ),
      map_manager_( map ),
      tracker_( tracker ),
      viz_( viz ) {
  use_clahe_ = param_->feat_tracker_setting_.use_clahe_;

  if ( use_clahe_ ) {
    int tilesize = 50;
    cv::Size clahe_tiles( param_->cam_setting_.left_resolution_[0] / tilesize,
                          param_->cam_setting_.left_resolution_[1] / tilesize );
    clahe_ = cv::createCLAHE( param_->feat_tracker_setting_.clahe_val_, clahe_tiles );
  }
  klt_use_prior_ = param_->feat_tracker_setting_.klt_use_prior_;
  klt_win_size_ = param_->feat_tracker_setting_.klt_win_size_;
  klt_pyr_level_ = param_->feat_tracker_setting_.klt_pyr_level_;
  klt_err_ = param_->feat_tracker_setting_.klt_err_;
  klt_max_fb_dist_ = param_->feat_tracker_setting_.klt_max_fb_dist_;
  track_keyframetoframe_ = param_->feat_tracker_setting_.track_keyframetoframe_;
}

bool VisualFrontEnd::VisualTracking( cv::Mat& image, double time ) {
  std::lock_guard<std::mutex> lock( map_manager_->map_mutex_ );
  
  bool is_kf = TrackerMono( image, time );
  if ( is_kf ) {
    map_manager_->CreateKeyframe( cur_img_, image );
  }

  return is_kf;
}

bool VisualFrontEnd::TrackerMono( cv::Mat& image, double time ) {
  // preprocess
  PreProcessImage( image );

  // first frame is keyframe
  if ( current_frame_->id_ == 0 ) {
    return true;
  }

  // Sophus::SE3d Twc = current_frame_->GetTwc();
  // motion_model_.applyMotionModel( Twc, time );
  // current_frame_->SetTwc( Twc );

  // tracking from frame to frame
  KLTTracking();

  // outlier filter
  Epipolar2d2dFiltering();

  // show tracking result to ui
  ShowTrackingResult();

  // compute current visual frontend pose
  ComputePose();

  // update motion model
  UpdateMotionModel( time );

  // check is new keyframe
  return CheckIsNewKeyframe();
}

void VisualFrontEnd::PreProcessImage( cv::Mat& img_raw ) {
  if ( use_clahe_ ) {
    clahe_->apply( img_raw, cur_img_ );
  } else {
    cur_img_ = img_raw;
  }

  if ( !cur_pyr_.empty() && !track_keyframetoframe_ ) {
    prev_pyr_.swap( cur_pyr_ );
  }

  cv::buildOpticalFlowPyramid( cur_img_, cur_pyr_, cv::Size( klt_win_size_, klt_win_size_ ),
                               param_->feat_tracker_setting_.klt_pyr_level_ );
}

void VisualFrontEnd::KLTTracking() {
  // Get current kps and init priors for tracking
  std::vector<int> v3d_kp_ids, vkp_ids;
  std::vector<cv::Point2f> v3d_kps, v3d_priors, vkps, vpriors;
  std::vector<bool> vkp_is3d;

  // First we're gonna track 3d kps on only 2 levels
  v3d_kp_ids.reserve( current_frame_->nb3dkps_ );
  v3d_kps.reserve( current_frame_->nb3dkps_ );
  v3d_priors.reserve( current_frame_->nb3dkps_ );

  // Then we'll track 2d kps on full pyramid levels
  vkp_ids.reserve( current_frame_->nbkps_ );
  vkps.reserve( current_frame_->nbkps_ );
  vpriors.reserve( current_frame_->nbkps_ );

  vkp_is3d.reserve( current_frame_->nbkps_ );

  // Front-End is thread-safe so we can direclty access curframe's kps
  for ( const auto& it : current_frame_->mapkps_ ) {
    auto& kp = it.second;

    // Init prior px pos. from motion model
    if ( klt_use_prior_ ) {
      if ( kp.is3d_ ) {
        // TODO
      }
    }

    // For other kps init prior with prev px pos.
    vkp_ids.push_back( kp.lmid_ );
    vkps.push_back( kp.px_ );
    vpriors.push_back( kp.px_ );
  }

  // 1st track 3d kps if using prior
  if ( klt_use_prior_ && !v3d_priors.empty() ) {
    // TODO
  }
  // 2st tracker 2d kps
  if ( !vkps.empty() ) {
    // Good / bad kps vector
    std::vector<bool> kpstatus;

    tracker_->trackerAndMatcher( prev_pyr_, cur_pyr_, klt_win_size_, klt_pyr_level_, klt_err_,
                                 klt_max_fb_dist_, vkps, vpriors, kpstatus );

    size_t good_num = 0;

    for ( size_t i = 0; i < vkps.size(); i++ ) {
      if ( kpstatus.at( i ) ) {
        current_frame_->UpdateKeypoint( vkp_ids.at( i ), vpriors.at( i ) );
        good_num++;
      } else {
        // MapManager is responsible for all the removing operations
        map_manager_->RemoveObsFromCurFrameById( vkp_ids.at( i ) );
      }
    }
  }
}

void VisualFrontEnd::Epipolar2d2dFiltering() {
  // Get prev KF
  auto pkf = map_manager_->GetKeyframe( current_frame_->kfid_ );

  if ( pkf == nullptr ) {
    LOG( FATAL ) << "ERROR! Previous Kf does not exist yet (epipolar2d2d()).";
  }

  // Get cur. Frame nb kps
  size_t nbkps = current_frame_->nbkps_;

  if ( nbkps < 8 ) {
    LOG( WARNING ) << "Not enough kps to compute Essential Matrix";
    return;
  }
  std::vector<int> vkpsids;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
  std::vector<cv::Point2f> vkf_px, vcur_px;

  for ( const auto& it : current_frame_->mapkps_ ) {
    auto& kp = it.second;

    // Get the prev. KF related kp if it exists
    auto kf_kp = pkf->GetKeypointById( kp.lmid_ );

    if ( kf_kp.lmid_ != kp.lmid_ ) {
      continue;
    }
    vkfbvs.push_back( kf_kp.bv_ );
    vcurbvs.push_back( kp.bv_ );
    vkf_px.push_back( kf_kp.px_ );
    vcur_px.push_back( kp.px_ );
    vkpsids.push_back( kp.lmid_ );
    // TODO
  }
  std::vector<uchar> inliers;
  cv::findFundamentalMat( vkf_px, vcur_px, cv::FM_RANSAC, 3, 0.99, inliers );
  assert( vkf_px.size() == vcur_px.size() && vcur_px.size() == inliers.size() );
  for ( size_t i = 0; i < inliers.size(); i++ ) {
    if ( !inliers[i] ) {
      map_manager_->RemoveObsFromCurFrameById( vkpsids[i] );
    }
  }
}

void VisualFrontEnd::ShowTrackingResult() {
  cv::cvtColor( cur_img_, draw_tracker_image_, cv::COLOR_GRAY2BGR );
  // Get prev KF
  auto pkf = map_manager_->GetKeyframe( current_frame_->kfid_ );
  std::vector<cv::Point2f> vkf_px, vcur_px;
  for ( const auto& it : current_frame_->mapkps_ ) {
    auto& kp = it.second;
    // Get the prev. KF related kp if it exists
    auto kf_kp = pkf->GetKeypointById( kp.lmid_ );
    if ( kf_kp.lmid_ != kp.lmid_ ) {
      continue;
    }
    vkf_px.push_back( kf_kp.px_ );
    vcur_px.push_back( kp.px_ );
    cv::arrowedLine( draw_tracker_image_, kp.px_, kf_kp.px_, cv::Scalar( 0, 255, 0 ), 2, 8, 0,
                     0.3 );
    cv::circle( draw_tracker_image_, kf_kp.px_, 2, cv::Scalar( 0, 255, 0 ), -1 );
  }
}

bool VisualFrontEnd::CheckIsNewKeyframe() {
  // Get prev. KF
  auto pkf = map_manager_->GetKeyframe( current_frame_->kfid_ );
  if ( pkf == nullptr ) {
    LOG( FATAL ) << "Cant get prev keyframe.";
  }

  // id diff with last KF
  int num_img_from_kf = current_frame_->id_ - pkf->id_;

  // 3d keypoint number
  if ( current_frame_->nb3dkps_ < 20 ) {
    return true;
  }

  float parallax = ComputeParallax( pkf->kfid_, true, true, false );

  double timer_differ = current_frame_->img_time_ - pkf->img_time_;

  if ( param_->slam_setting_.stereo_mode_ && timer_differ > 1. ) {
    return true;
  }

  return true;
}

float VisualFrontEnd::ComputeParallax( const int kfid, bool do_un_rot, bool median,
                                       bool is_2donly ) {
  // Get prev. KF
  auto kf = map_manager_->GetKeyframe( kfid );
  if ( kf == nullptr ) {
    LOG( WARNING ) << "[Visual Front End] Error in computeParallax ! Prev KF #" << kfid
                   << " does not exist!";
    return 0.;
  }

  // Compute relative rotation between cur Frame and prev. KF if required
  Eigen::Matrix3d R_kf_cur( Eigen::Matrix3d::Identity() );
  if ( do_un_rot ) {
    Eigen::Matrix3d Rkfw = kf->GetTcw().rotationMatrix();
    Eigen::Matrix3d Rwcur = current_frame_->GetTwc().rotationMatrix();
    R_kf_cur = Rkfw * Rwcur;
  }

  // Compute parallax
  float avg_parallax = 0.;
  int num_parallax = 0;
  std::set<float> set_parallax;

  for ( const auto& it : current_frame_->mapkps_ ) {
    if ( is_2donly && it.second.is3d_ ) {
      continue;
    }
    auto& kp = it.second;
    // Get prev. KF kp if it exists
    auto kfkp = kf->GetKeypointById( kp.lmid_ );

    if ( kfkp.lmid_ != kp.lmid_ ) {
      continue;
    }

    // Compute parallax with unpx pos.
    cv::Point2f unpx = kp.unpx_;

    // Rotate bv into KF cam frame and back project into image
    if ( do_un_rot ) {
      unpx = kf->ProjCamToImage( R_kf_cur * kp.bv_ );
    }

    // Compute rotation-compensated parallax
    float parallax = cv::norm( unpx - kfkp.unpx_ );
    avg_parallax += parallax;
    num_parallax++;

    if ( median ) {
      set_parallax.insert( parallax );
    }
  }

  if ( num_parallax == 0 ) {
    return 0.;
  }

  // Average parallax
  avg_parallax /= static_cast<float>( num_parallax );

  if ( median ) {
    auto it = set_parallax.begin();
    std::advance( it, set_parallax.size() / 2 );
    avg_parallax = *it;
  }

  return avg_parallax;
}

void VisualFrontEnd::ComputePose() {
  // Get cur nb of 3D kps
  size_t nb3dkps = current_frame_->nb3dkps_;
  if ( nb3dkps < 4 ) {
    LOG( WARNING ) << ">>> Not enough kps to compute P3P / PnP";
    return;
  }
  // Setup P3P-Ransac computation for OpenGV-based Pose estimation + motion-only BA with Ceres
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts, vbvs;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > vkps;
  std::vector<int> vkpids, voutliersidx;

  for ( const auto& it : current_frame_->mapkps_ ) {
    if ( !it.second.is3d_ ) {
      continue;
    }
    auto& kp = it.second;
    auto plm = map_manager_->GetMapPoint( kp.lmid_ );
    if ( plm == nullptr ) {
      continue;
    }
    // for P3P
    vbvs.push_back( kp.bv_ );

    vkps.push_back( Eigen::Vector2d( kp.unpx_.x, kp.unpx_.y ) );
    vwpts.push_back( plm->GetPoint() );
    vkpids.push_back( kp.lmid_ );
  }

  std::vector<int> vscales( vkps.size(), 0 );
  Sophus::SE3d Twc = current_frame_->GetTwc();
  bool success = false;
  int max_iters = 5;
  float robust_mono_th = 5.9915;
  bool use_robust = true;
  Eigen::Matrix3d K = current_frame_->pcalib_leftcam_->K_;
  success = geometry::tceresMotionOnlyBA( vkps, vwpts, vscales, Twc, max_iters, robust_mono_th,
                                          use_robust, true, K, voutliersidx );

  // success = geometry::opencvP3PRansac( vbvs, vwpts, 100, 3., K( 0, 0 ), K( 1, 1 ), true, Twc,
  //                                      voutliersidx );

  // Check that pose estim. was good enough
  size_t nbinliers = vwpts.size() - voutliersidx.size();
  if ( !success || nbinliers < 5 || voutliersidx.size() > 0.5 * vwpts.size() ||
       Twc.translation().array().isInf().any() || Twc.translation().array().isNaN().any() ) {
    LOG( WARNING ) << "ceres/OpenCV PNP calculate " << ( success ? "success" : "failed" )
                   << " num inliers " << nbinliers << ", num outliers " << voutliersidx.size()
                   << ", vwpts.size: " << vwpts.size();
  }
  // Update frame pose
  current_frame_->SetTwc( Twc );
  // Remove outliers
  for ( const auto& idx : voutliersidx ) {
    // MapManager is responsible for all removing operations
    map_manager_->RemoveObsFromCurFrameById( vkpids.at( idx ) );
  }

}

void VisualFrontEnd::UpdateMotionModel( double time ) {
  motion_model_.updateMotionModel( current_frame_->GetTwc(), time );
}

std::vector<cv::Mat> VisualFrontEnd::GetCurrentFramePyramid() const { return cur_pyr_; }
cv::Mat VisualFrontEnd::GetTrackResultImage() const { return draw_tracker_image_; }

}  // namespace viohw
