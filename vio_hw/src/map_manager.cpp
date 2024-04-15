#include "vio_hw/internal/map_manager.hpp"

namespace viohw {

MapManager::MapManager( SettingPtr state, FramePtr frame, FeatureBasePtr feat_extract,
                        TrackerBasePtr tracker )
    : param_( state ),
      current_frame_( frame ),
      feature_extractor_( feat_extract ),
      tracker_( tracker ),
      lm_id_( 0 ),
      kf_id_( 0 ),
      num_lms_( 0 ),
      num_kfs_( 0 ) {}

void MapManager::CreateKeyframe( const cv::Mat& im, const cv::Mat& im_raw ) {
  PrepareFrame();
  ExtractKeypoints( im, im_raw );
  AddKeyframe();
}

void MapManager::PrepareFrame() {
  // Update new KF id
  current_frame_->kfid_ = kf_id_;

  for ( const auto& kp : current_frame_->GetKeypoints() ) {
    // Get the related MP
    auto lm_iter = map_lms_.find( kp.lmid_ );

    if ( lm_iter == map_lms_.end() ) {
      RemoveObsFromCurFrameById( kp.lmid_ );
      continue;
    }

    // Relate new KF id to the MP
    lm_iter->second->AddKfObs( kf_id_ );
  }
}

void MapManager::AddKeyframe() {  // Create a copy of Cur. Frame shared_ptr for creating an
  // independant KF to add to the map
  std::shared_ptr<Frame> pkf =
      std::allocate_shared<Frame>( Eigen::aligned_allocator<Frame>(), *current_frame_ );

  std::lock_guard<std::mutex> lock( kf_mutex_ );

  // Add KF to the unordered map and update id/nb
  map_kfs_.emplace( kf_id_, pkf );
  // num_kfs_++;
  NumKFPlus();
  kf_id_++;
}

void MapManager::DescribeKeypoints( const cv::Mat& im, const std::vector<Keypoint>& vkps,
                                    const std::vector<cv::Point2f>& vpts ) {
  size_t nbkps = vkps.size();
  std::vector<cv::Mat> vdescs;

  vdescs = feature_extractor_->DescribeBRIEF( im, vpts );

  assert( vkps.size() == vdescs.size() );

  for ( size_t i = 0; i < nbkps; i++ ) {
    if ( !vdescs.at( i ).empty() ) {
      current_frame_->UpdateKeypointDesc( vkps.at( i ).lmid_, vdescs.at( i ) );
      map_lms_.at( vkps.at( i ).lmid_ )->AddDesc( current_frame_->kfid_, vdescs.at( i ) );
    }
  }
}

void MapManager::ExtractKeypoints( const cv::Mat& im, const cv::Mat& im_raw ) {
  // get current kps which have been tracked, the location is changed in current frame
  // we need to update the desc in current location
  std::vector<viohw::Keypoint> kps = current_frame_->GetKeypoints();

  // TODO brief calculate
  if ( param_->feat_tracker_setting_.use_brief_ ) {
    //.................
    std::vector<cv::Point2f> vpts;
    for ( auto& kp : kps ) {
      vpts.push_back( kp.px_ );
    }
    DescribeKeypoints( im_raw, kps, vpts );
  }

  cv::Mat mask = cv::Mat( im.rows, im.cols, CV_8UC1, cv::Scalar( 255 ) );
  for ( auto& pt : kps ) {
    cv::circle( mask, pt.px_, param_->feat_tracker_setting_.max_feature_dis_, 0, -1 );
    // cv::rectangle( mask, pt.px_ - cv::Point2f( 10, 10 ), pt.px_ + cv::Point2f( 10, 10 ), 0,
    //                cv::FILLED );
  }

  int num_need_detect = param_->feat_tracker_setting_.max_feature_num_ - kps.size();
  if ( num_need_detect > 0 ) {
    std::vector<cv::KeyPoint> new_kps;
    Eigen::Matrix<double, 259, Eigen::Dynamic> feat;
    cv::Mat desc;
    feature_extractor_->setTobeExtractKpsNumber( num_need_detect );
    feature_extractor_->detect( im, new_kps, mask, desc, feat );
    if ( !new_kps.empty() ) {
      std::vector<cv::Point2f> desc_pts;
      cv::KeyPoint::convert( new_kps, desc_pts );
      std::vector<cv::Mat> vdescs = feature_extractor_->DescribeBRIEF( im_raw, desc_pts );
      if ( param_->feat_tracker_setting_.tracker_type_ == TrackerBase::LIGHT_GLUE ) {
        AddKeypointsToFrame( desc_pts, vdescs, feat, *current_frame_ );
      } else {
        AddKeypointsToFrame( desc_pts, vdescs, *current_frame_ );
      }
    }
  }
}

void MapManager::AddKeypointsToFrame( const std::vector<cv::Point2f>& vpts, Frame& frame ) {
  std::lock_guard<std::mutex> lock( lm_mutex_ );

  // Add keypoints + create MPs
  for ( const auto& vpt : vpts ) {
    // Add keypoint to current frame
    frame.AddKeypoint( vpt, lm_id_ );
    // Create landmark with same id
    AddMapPoint();
  }
}

void MapManager::AddKeypointsToFrame( const std::vector<cv::Point2f>& vpts,
                                      const std::vector<cv::Mat>& vdescs,
                                      const Eigen::Matrix<double, 259, Eigen::Dynamic>& feat,
                                      viohw::Frame& frame ) {
  CHECK( vpts.size() == feat.cols() );
  CHECK( vdescs.size() == feat.cols() );
  std::lock_guard<std::mutex> lock( lm_mutex_ );
  // Add keypoints + create landmarks
  int num = 0;
  for ( size_t i = 0; i < vpts.size(); i++ ) {
    if ( !vdescs.at( i ).empty() ) {
      // Add keypoint to current frame
      frame.AddKeypoint( vpts.at( i ), lm_id_, vdescs.at( i ), feat.col( i ) );
      // Create landmark with same id
      AddMapPoint( vdescs.at( i ) );
    } else {
      // Add keypoint to current frame
      frame.AddKeypoint( vpts.at( i ), lm_id_ );
      // Create landmark with same id
      AddMapPoint();
    }
  }
}

void MapManager::AddKeypointsToFrame( const std::vector<cv::Point2f>& vpts,
                                      const std::vector<cv::Mat>& vdescs, Frame& frame ) {
  std::lock_guard<std::mutex> lock( lm_mutex_ );

  // Add keypoints + create landmarks
  for ( size_t i = 0; i < vpts.size(); i++ ) {
    if ( !vdescs.at( i ).empty() ) {
      // Add keypoint to current frame
      frame.AddKeypoint( vpts.at( i ), lm_id_, vdescs.at( i ) );
      // Create landmark with same id
      AddMapPoint( vdescs.at( i ) );
    } else {
      // Add keypoint to current frame
      frame.AddKeypoint( vpts.at( i ), lm_id_ );
      // Create landmark with same id
      AddMapPoint();
    }
  }
}

void MapManager::AddMapPoint() {
  // Create a new MP with a unique lmid and a KF id obs
  std::shared_ptr<MapPoint> plm =
      std::allocate_shared<MapPoint>( Eigen::aligned_allocator<MapPoint>(), lm_id_, kf_id_ );

  // Add new MP to the map and update id/nb
  map_lms_.emplace( lm_id_, plm );
  lm_id_++;
  // num_lms_++;
  NumLandmarkPlus();
  // Visualization related part for pointcloud obs
  // TODO
}

void MapManager::AddMapPoint( const cv::Mat& desc ) {
  // Create a new MP with a unique lmid and a KF id obs
  std::shared_ptr<MapPoint> plm =
      std::allocate_shared<MapPoint>( Eigen::aligned_allocator<MapPoint>(), lm_id_, kf_id_, desc );

  // Add new MP to the map and update id/nb
  map_lms_.emplace( lm_id_, plm );
  lm_id_++;
  // num_lms_++;
  NumLandmarkPlus();
  // Visualization related part for pointcloud obs
  // TODO
}

// Remove a MP obs from cur Frame
void MapManager::RemoveObsFromCurFrameById( const int lmid ) {
  // Remove cur obs
  current_frame_->RemoveKeypointById( lmid );

  // Set MP as not obs
  // TODO
}

std::shared_ptr<Frame> MapManager::GetKeyframe( const int kfid ) const {
  std::lock_guard<std::mutex> lock( kf_mutex_ );

  auto it = map_kfs_.find( kfid );
  if ( it == map_kfs_.end() ) {
    return nullptr;
  }
  return it->second;
}

std::shared_ptr<MapPoint> MapManager::GetMapPoint( const int lmid ) const {
  std::lock_guard<std::mutex> lock( lm_mutex_ );

  auto it = map_lms_.find( lmid );
  if ( it == map_lms_.end() ) {
    return nullptr;
  }
  return it->second;
}

void MapManager::StereoMatching( Frame& frame, const std::vector<cv::Mat>& vleftpyr,
                                 const std::vector<cv::Mat>& vrightpyr ) {
  // Find stereo correspondances with left kps
  auto vleftkps = frame.GetKeypoints();
  size_t nbkps = vleftkps.size();

  std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
  std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
  Eigen::Matrix<double, 259, Eigen::Dynamic> feat_prev, feat_cur;
  feat_prev.resize( 259, static_cast<Eigen::Index>( nbkps ) );

  for ( size_t i = 0; i < nbkps; i++ ) {
    // Set left kp
    auto& kp = vleftkps.at( i );

    // Set prior right kp
    cv::Point2f priorpt = kp.px_;

    vkpids.push_back( kp.lmid_ );
    vkps.push_back( kp.px_ );
    vpriors.push_back( priorpt );
    feat_prev.col( i ) = kp.sp_feat_desc_;
  }

  std::vector<cv::Point2f> good_right_kps;
  std::vector<int> good_ids;
  size_t num_good = 0, tracker_good = 0, inliner_good = 0;
  if ( !vkps.empty() ) {
    // Good / bad kps vector
    std::vector<bool> vkpstatus;
    std::vector<uchar> inliers;
    tracker_->trackerAndMatcher( vleftpyr, vrightpyr, vkps, vpriors, vkpstatus, feat_prev,
                                 feat_cur );

    for ( size_t i = 0; i < vkpstatus.size(); i++ ) {
      if ( vkpstatus.at( i ) ) {
        frame.UpdateKeypointStereo( vkpids.at( i ), vpriors.at( i ) );
        num_good++;
      }
    }
    // TODO
  }
  // LOG(INFO) << "kp num: " << vkps.size() << ",Good Stereo Matching Num: " << num_good
  //           << ",tracker good:" << tracker_good << ", inliner good:" << inliner_good;
}

void MapManager::UpdateMapPoint( const int lmid, const Eigen::Vector3d& wpt,
                                 const double inv_depth ) {
  std::lock_guard<std::mutex> lock( lm_mutex_ );
  std::lock_guard<std::mutex> lock_kf( kf_mutex_ );
  auto plmit = map_lms_.find( lmid );

  if ( plmit == map_lms_.end() ) {
    return;
  }
  if ( plmit->second == nullptr ) {
    return;
  }
  // If MP 2D -> 3D => Notif. KFs
  if ( !plmit->second->is3d_ ) {
    for ( const auto& kfid : plmit->second->GetKfObsSet() ) {
      auto pkfit = map_kfs_.find( kfid );
      if ( pkfit != map_kfs_.end() ) {
        pkfit->second->TurnKeypoint3d( lmid );
      } else {
        plmit->second->RemoveKfObs( kfid );
      }
    }
    if ( plmit->second->isobs_ ) {
      current_frame_->TurnKeypoint3d( lmid );
    }
  }

  // Update MP world pos.
  if ( inv_depth >= 0. ) {
    plmit->second->SetPoint( wpt, inv_depth );
  } else {
    plmit->second->SetPoint( wpt );
  }
}
// Remove a MP from the map
void MapManager::RemoveMapPoint( const int lmid ) {
  std::lock_guard<std::mutex> lock( lm_mutex_ );
  std::lock_guard<std::mutex> lockkf( kf_mutex_ );

  // Get related MP
  auto plmit = map_lms_.find( lmid );
  // Skip if MP does not exist
  if ( plmit != map_lms_.end() ) {
    // Remove all observations from KFs
    for ( const auto& kfid : plmit->second->GetKfObsSet() ) {
      auto pkfit = map_kfs_.find( kfid );
      if ( pkfit == map_kfs_.end() ) {
        continue;
      }
      pkfit->second->RemoveKeypointById( lmid );

      for ( const auto& cokfid : plmit->second->GetKfObsSet() ) {
        if ( cokfid != kfid ) {
          // TODO
          // pkfit->second->decreaseCovisibleKf(cokfid);
        }
      }
    }

    // If obs in cur Frame, remove cur obs
    if ( plmit->second->isobs_ ) {
      current_frame_->RemoveKeypointById( lmid );
    }

    if ( plmit->second->is3d_ ) {
      num_lms_--;
    }

    // Erase MP and update nb MPs
    map_lms_.erase( plmit );
  }

  // Visualization related part for pointcloud obs
  // TODO
}

// Remove a KF obs from a MP
void MapManager::RemoveMapPointObs( const int lmid, const int kfid ) {
  std::lock_guard<std::mutex> lock( lm_mutex_ );
  std::lock_guard<std::mutex> lockkf( kf_mutex_ );

  // Remove MP obs from KF
  auto pkfit = map_kfs_.find( kfid );
  if ( pkfit != map_kfs_.end() ) {
    pkfit->second->RemoveKeypointById( lmid );
  }

  // Remove KF obs from MP
  auto plmit = map_lms_.find( lmid );

  // Skip if MP does not exist
  if ( plmit == map_lms_.end() ) {
    return;
  }
  plmit->second->RemoveKfObs( kfid );

  // TODO
  // if( pkfit != map_pkfs_.end() ) {
  //   for( const auto &cokfid : plmit->second->getKfObsSet() ) {
  //     auto pcokfit = map_pkfs_.find(cokfid);
  //     if( pcokfit != map_pkfs_.end() ) {
  //       pkfit->second->decreaseCovisibleKf(cokfid);
  //       pcokfit->second->decreaseCovisibleKf(kfid);
  //     }
  //   }
  // }
}

int MapManager::GetNumberKF() const {
  std::lock_guard<std::mutex> lck( num_kfs_mutex_ );
  return num_kfs_;
}
void MapManager::NumKFPlus() {
  std::lock_guard<std::mutex> lck( num_kfs_mutex_ );
  num_kfs_++;
}
int MapManager::GetNumberLandmark() const {
  std::lock_guard<std::mutex> lck( num_lms_mutex_ );
  return num_lms_;
}
void MapManager::NumLandmarkPlus() {
  std::lock_guard<std::mutex> lck( num_lms_mutex_ );
  num_lms_++;
}
}  // namespace viohw
