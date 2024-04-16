#include "vio_hw/internal/mapping.hpp"

namespace viohw {

Mapping::Mapping( viohw::SettingPtr param, viohw::MapManagerPtr map_manager, viohw::FramePtr frame,
                  LoopCloserPtr loop, EstimatorPtr estimator )
    : params_( param ),
      map_manager_( map_manager ),
      current_frame_( frame ),
      loop_closer_( loop ),
      estimator_( estimator ) {
  stereo_mode_ = params_->slam_setting_.stereo_mode_;
  use_clahe_ = params_->feat_tracker_setting_.use_clahe_;
  use_loop_ = params_->loop_setting_.use_loop_closer_;

  if ( use_clahe_ ) {
    int tilesize = 50;
    cv::Size clahe_tiles( params_->cam_setting_.left_resolution_[0] / tilesize,
                          params_->cam_setting_.left_resolution_[1] / tilesize );
    clahe_ = cv::createCLAHE( params_->feat_tracker_setting_.clahe_val_, clahe_tiles );
  }
  std::thread mapper_thread( &Mapping::run, this );
  mapper_thread.detach();
}

void Mapping::run() {
  std::thread lc_thread( &LoopCloser::run, loop_closer_ );
  std::thread estimator_thread( &Estimator::run, estimator_ );

  int klt_win_size = params_->feat_tracker_setting_.klt_win_size_;
  cv::Size cv_klt_win_size( klt_win_size, klt_win_size );
  int klt_pyra_level = params_->feat_tracker_setting_.klt_pyr_level_;
  bool open_backend_opt = params_->backend_optimization_setting_.open_backend_opt_;

  Keyframe kf;

  while ( true ) {
    if ( GetNewKf( kf ) ) {
      // Get new KF ptr
      std::shared_ptr<Frame> new_kf = map_manager_->GetKeyframe( kf.kfid_ );
      assert( new_kf );
      if ( stereo_mode_ ) {
        cv::Mat img_right, img_left = kf.imleftraw_;
        if ( use_clahe_ ) {
          clahe_->apply( kf.imrightraw_, img_right );
        } else {
          img_right = kf.imrightraw_;
        }
        std::vector<cv::Mat> vpyr_img_right;
        cv::buildOpticalFlowPyramid( img_right, vpyr_img_right, cv_klt_win_size, klt_pyra_level );
        map_manager_->StereoMatching( *new_kf, kf.vpyr_imleft_, vpyr_img_right );
        if ( new_kf->nb2dkps_ > 0 && new_kf->nb_stereo_kps_ > 0 ) {
          std::lock_guard<std::mutex> lock( map_manager_->map_mutex_ );
          TriangulateStereo( *new_kf );
        }
      }
      if ( new_kf->nb2dkps_ > 0 && new_kf->kfid_ > 0 ) {
        std::lock_guard<std::mutex> lock( map_manager_->map_mutex_ );
        TriangulateTemporal( *new_kf );
      }

      if ( open_backend_opt ) {
        estimator_->AddNewKf( new_kf );
      }
      if ( use_loop_ ) {
        loop_closer_->AddNewKeyFrame( new_kf, kf.imleftraw_ );
      }

    } else {
      std::chrono::microseconds dura( 100 );
      std::this_thread::sleep_for( dura );
    }
  }
}

void Mapping::TriangulateTemporal( Frame& frame ) {
  // Get New KF kps / pose
  std::vector<Keypoint> vkps = frame.GetKeypoints2d();
  size_t nbkps = vkps.size();

  Sophus::SE3d Twcj = frame.GetTwc();
  if ( vkps.empty() ) {
    LOG( WARNING ) << ">>> No kps to temporal triangulate...";
    return;
  }

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vleftbvs, vrightbvs;

  // Relative motions between new KF and prev. KFs
  int relkfid = -1;
  Sophus::SE3d Tcicj, Tcjci;
  Eigen::Matrix3d Rcicj;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vwpts;
  std::vector<int> vlmids;
  int good = 0;

  for ( size_t i = 0; i < nbkps; i++ ) {
    // Get the related MP and check if it is ready to be triangulated
    std::shared_ptr<MapPoint> plm = map_manager_->GetMapPoint( vkps.at( i ).lmid_ );
    if ( plm == nullptr ) {
      map_manager_->RemoveMapPointObs( vkps.at( i ).lmid_, frame.kfid_ );
      continue;
    }
    // If MP is already 3D continue (should not happen)
    if ( plm->is3d_ ) {
      continue;
    }
    // Get the set of KFs sharing observation of this 2D MP
    std::set<int> co_kf_ids = plm->GetKfObsSet();
    // Continue if new KF is the only one observing it
    if ( co_kf_ids.size() < 2 ) {
      continue;
    }

    int kfid = *co_kf_ids.begin();
    if ( frame.kfid_ == kfid ) {
      continue;
    }

    // Get the 1st KF observation of the related MP
    auto pkf = map_manager_->GetKeyframe( kfid );

    if ( pkf == nullptr ) {
      continue;
    }

    // Compute relative motion between new KF and selected KF (only if req.)
    if ( relkfid != kfid ) {
      Sophus::SE3d Tciw = pkf->GetTcw();
      Tcicj = Tciw * Twcj;
      Tcjci = Tcicj.inverse();
      Rcicj = Tcicj.rotationMatrix();

      relkfid = kfid;
    }
    // If no motion between both KF, skip
    if ( params_->slam_setting_.stereo_mode_ && Tcicj.translation().norm() < 0.01 ) {
      continue;
    }

    // TODO if using imu in mono mode, the [Tcicj.translation().norm() < 0.01] also can be use
    // If no motion between both KF, skip
    // if ( !params_->slam_setting_.stereo_mode_ && params_->slam_setting_.use_imu_ &&
    //      Tcicj.translation().norm() < 0.01 ) {
    //   continue;
    // }

    // Get obs kp
    Keypoint kfkp = pkf->GetKeypointById( vkps.at( i ).lmid_ );
    if ( kfkp.lmid_ != vkps.at( i ).lmid_ ) {
      continue;
    }

    // Check rotation-compensated parallax
    cv::Point2f rotpx = frame.ProjCamToImage( Rcicj * vkps.at( i ).bv_ );
    double parallax = cv::norm( kfkp.unpx_ - rotpx );
    // Compute 3D pos and check if its good or not
    Eigen::Vector3d left_pt = ComputeTriangulation( Tcicj, kfkp.bv_, vkps.at( i ).bv_ );
    // Project into right cam (new KF)
    Eigen::Vector3d right_pt = Tcjci * left_pt;
    // Ensure that the 3D MP is in front of both camera
    if ( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
      if ( parallax > 20. ) {
        map_manager_->RemoveMapPointObs( kfkp.lmid_, frame.kfid_ );
      }
      continue;
    }

    // Remove MP with high reprojection error
    cv::Point2f left_px_proj = pkf->ProjCamToImage( left_pt );
    cv::Point2f right_px_proj = frame.ProjCamToImage( right_pt );
    double ldist = cv::norm( left_px_proj - kfkp.unpx_ );
    double rdist = cv::norm( right_px_proj - vkps.at( i ).unpx_ );

    if ( ldist > 30. || rdist > 30. ) {
      if ( parallax > 20. ) {
        map_manager_->RemoveMapPointObs( kfkp.lmid_, frame.kfid_ );
      }
      continue;
    }

    // The 3D pos is good, update SLAM MP and related KF / Frame
    Eigen::Vector3d wpt = pkf->ProjCamToWorld( left_pt );
    map_manager_->UpdateMapPoint( vkps.at( i ).lmid_, wpt, 1. / left_pt.z() );

    good++;
  }
  size_t total_kps = frame.nbkps_;
  size_t nb_stereo_kps = frame.nb_stereo_kps_;
  size_t nb_3d_kps = frame.nb3dkps_;
  size_t nb_2d_kps = frame.nb2dkps_;
  std::string output = tceres::internal::StringPrintf(
      "TriangulateTemporal success %d, "
      "total kps %d, stereo kps %d, 3d kps %d, 2d kps %d",
      good, total_kps, nb_stereo_kps, nb_3d_kps, nb_2d_kps );
  // LOG( INFO ) << output;
}

void Mapping::TriangulateStereo( Frame& frame ) {
  std::vector<Keypoint> vkps = frame.getKeypointsStereo();
  size_t num_kps = vkps.size();

  if ( vkps.empty() ) {
    LOG( WARNING ) << ">>> No kps to stereo triangulate...";
    return;
  }

  // Store the stereo kps along with their idx
  std::vector<int> stereo_idx;
  std::vector<int> lm_ids;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > left_bvs, right_bvs;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > world_pts;

  Sophus::SE3d Tlr = frame.pcalib_rightcam_->getExtrinsic();
  Sophus::SE3d Trl = Tlr.inverse();

  size_t num_already_3d_kps = 0;
  for ( size_t i = 0; i < num_kps; i++ ) {
    if ( vkps.at( i ).is3d_ ) {
      num_already_3d_kps++;
      continue;
    }

    if ( !vkps.at( i ).is3d_ && vkps.at( i ).is_stereo_ ) {
      stereo_idx.push_back( i );
      left_bvs.push_back( vkps.at( i ).bv_ );
      right_bvs.push_back( vkps.at( i ).rbv_ );
    }
  }

  if ( stereo_idx.empty() && num_already_3d_kps != num_kps ) {
    LOG( WARNING ) << "stereo feature is empty, num_already_3d_kps: " << num_already_3d_kps
                   << ", num_stereo_kps: " << num_kps;
    return;
  }

  int good = 0;

  for ( size_t i = 0; i < stereo_idx.size(); i++ ) {
    int kpidx = stereo_idx.at( i );
    Eigen::Vector3d left_pt = ComputeTriangulation( Tlr, left_bvs.at( i ), right_bvs.at( i ) );
    // Project into right cam frame
    Eigen::Vector3d right_pt = Trl * left_pt;
    if ( left_pt.z() < 0.1 || right_pt.z() < 0.1 ) {
      frame.RemoveStereoKeypointById( vkps.at( kpidx ).lmid_ );
      continue;
    }

    cv::Point2f left_px_proj = frame.ProjCamToImage( left_pt );
    cv::Point2f right_px_proj = frame.ProjCamToRightImage( left_pt );
    double ldist = cv::norm( left_px_proj - vkps.at( kpidx ).unpx_ );
    double rdist = cv::norm( right_px_proj - vkps.at( kpidx ).runpx_ );
    // TODO max_reprojection_error in params
    if ( ldist > 3. || rdist > 3. ) {
      frame.RemoveStereoKeypointById( vkps.at( kpidx ).lmid_ );
      continue;
    }

    // Project MP in world frame
    Eigen::Vector3d wpt = frame.ProjCamToWorld( left_pt );

    map_manager_->UpdateMapPoint( vkps.at( kpidx ).lmid_, wpt, 1. / left_pt.z() );

    good++;
  }
  size_t total_kps = frame.nbkps_;
  size_t nb_stereo_kps = frame.nb_stereo_kps_;
  size_t nb_3d_kps = frame.nb3dkps_;
  size_t nb_2d_kps = frame.nb2dkps_;
  std::string output = tceres::internal::StringPrintf(
      "TriangulateStereo success %d, "
      "total kps %d, stereo kps %d, 3d kps %d, 2d kps %d",
      good, total_kps, nb_stereo_kps, nb_3d_kps, nb_2d_kps );
  // LOG( INFO ) << output;
}

Eigen::Vector3d Mapping::ComputeTriangulation( const Sophus::SE3d& T, const Eigen::Vector3d& bvl,
                                               const Eigen::Vector3d& bvr ) {
  return geometry::OpencvTriangulate( T, bvl, bvr );
}

bool Mapping::GetNewKf( Keyframe& kf ) {
  std::lock_guard<std::mutex> lock( kf_queen_mutex_ );

  // Check if new KF is available
  if ( kfs_queen_.empty() ) {
    is_new_kf_available_ = false;
    return false;
  }

  // Get new KF and signal BA to stop if
  // it is still processing the previous KF
  kf = kfs_queen_.front();
  kfs_queen_.pop();

  // Setting is_new_kf_available_ to true will limit
  // the processing of the KF to triangulation and postpone
  // other costly tasks to next KF as we are running late!
  if ( kfs_queen_.empty() ) {
    is_new_kf_available_ = false;
  } else {
    is_new_kf_available_ = true;
  }

  return true;
}

void Mapping::AddNewKf( const Keyframe& kf ) {
  std::lock_guard<std::mutex> lock( kf_queen_mutex_ );

  kfs_queen_.push( kf );

  is_new_kf_available_ = true;
}

}  // namespace viohw
