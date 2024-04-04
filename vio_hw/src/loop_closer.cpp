#include "vio_hw/internal/loop_closer.hpp"

namespace viohw {

LoopCloser::LoopCloser( SettingPtr param, MapManagerPtr map_manager )
    : param_( param ), map_manager_( map_manager ) {
  use_loop_ = param->loop_setting_.use_loop_closer_;
  loop_threshold_ = param->loop_setting_.loop_threshold_;

  if ( use_loop_ ) {
    ORBVocabulary_ = std::make_unique<ORBVocabulary>();
    std::string voc_path = param_->config_file_path_setting_.vocabulary_path_;
    LOG( INFO ) << "open loop closing, read vocabulary txt file, wait a moment...";
    if ( voc_path.empty() || !ORBVocabulary_->loadFromTextFile( voc_path ) ) {
      LOG( FATAL ) << "Please check you vocabulary file path in the yaml config file";
    }
    fast_detect_ = cv::FastFeatureDetector::create( 20 );
    brief_cal_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
  } else {
    LOG( WARNING ) << ">>>>Loop Closing is close.>>>>";
  }
}

void LoopCloser::run() {
  if ( !use_loop_ ) {
    return;
  }

  while ( true ) {
    if ( GetNewKeyFrame() ) {
      if ( new_kf_ == nullptr ) {
        continue;
      }

      std::vector<cv::KeyPoint> cv_kps;
      cv::Mat cv_descs;
      ComputeDesc( cv_kps, cv_descs );
      if ( cv_descs.empty() ) {
        continue;
      }

      std::pair<int, float> candidate_kf_id_score = DetectLoop( cv_descs );

      if ( candidate_kf_id_score.second > 0.10 ) {
        // static int c = 0;
        // std::cout << GREEN << "Loop Closer Detect " << c++ << ", score: " << max_score
        //           << ". loop threshold is: " << loop_threshold_ << TAIL << std::endl;
        LOG( INFO ) << "Score: " << candidate_kf_id_score.second;
      } else {
        continue;
      }

      ProcessLoopCandidate( candidate_kf_id_score.first );

    } else {
      std::chrono::microseconds dura( 100 );
      std::this_thread::sleep_for( dura );
    }
  }
}

void LoopCloser::ProcessLoopCandidate( int kf_loop_id ) {
  // get loop kf
  auto lc_kf = map_manager_->GetKeyframe( kf_loop_id );
  if ( lc_kf == nullptr ) {
    LOG( WARNING ) << "loop kf is nullptr in the map";
  }

  // Pair of matched cur kp / map points
  std::vector<std::pair<int, int>> vkplmids;

  // Do a knnMatching to get a first set of matches
  KNNMatching( *new_kf_, *lc_kf, vkplmids );

  if ( vkplmids.size() < 15 ) {
    return;
  }

  std::vector<int> outliers_idx;
  bool epipolar_status = EpipolarFiltering( *new_kf_, *lc_kf, vkplmids, outliers_idx );

  size_t num_inliers = vkplmids.size() - outliers_idx.size();

  if ( !epipolar_status || num_inliers < 10 ) {
    LOG( WARNING ) << "Not enough inliers for LC after epipolar filtering";
    return;
  }

  if ( !outliers_idx.empty() ) {
    // Remove outliers from vector of pairs
    RemoveOutliers( vkplmids, outliers_idx );
  }

  Sophus::SE3d Twc = new_kf_->GetTwc();
  bool success = ComputePnP( *new_kf_, vkplmids, Twc, outliers_idx );

  num_inliers = vkplmids.size() - outliers_idx.size();
  if ( !success || num_inliers < 30 ) {
    LOG( WARNING ) << "Not enough inliers for LC after ComputePnP(). PNP "
                   << ( success ? "success" : "failed" ) << ", num_inliers: " << num_inliers;
    return;
  }
  LOG( INFO ) << "vkplmids: " << vkplmids.size() << ", num_inliers:" << num_inliers;

  if ( !outliers_idx.empty() ) {
    // Remove outliers from vector of pairs
    RemoveOutliers( vkplmids, outliers_idx );
  }

  size_t num_good_kps = vkplmids.size();
  if ( num_good_kps >= 30 ) {
    double lc_pose_err = ( new_kf_->GetTcw() * Twc ).log().norm();

    LOG( INFO ) << "[PoseGraph] >>> Closing a loop between : "
                << " KF #" << new_kf_->kfid_ << " (img #" << new_kf_->id_ << ") and KF #"
                << lc_kf->kfid_ << " (img #" << lc_kf->id_ << " ), lc pose err: " << lc_pose_err;

    LOG( INFO ) << "loop kf pos: " << lc_kf->GetTwc().translation().transpose() << "; "
                << "new  kf pos: " << new_kf_->GetTwc().translation().transpose() << "; "
                << "correct pos: " << Twc.translation().transpose();
  }
}

bool LoopCloser::ComputePnP( const Frame& frame, const std::vector<std::pair<int, int>>& vkplmids,
                             Sophus::SE3d& Twc, std::vector<int>& voutlier_idx ) {
  // Init vector for PnP
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vwpts, vbvs;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vkps;
  std::vector<int> vgoodkpidx, vscales, voutidx;

  size_t nbkps = vkplmids.size();

  vgoodkpidx.reserve( nbkps );
  vkps.reserve( nbkps );
  vwpts.reserve( nbkps );
  vscales.reserve( nbkps );
  voutidx.reserve( nbkps );

  // Get kps & MPs
  for ( size_t i = 0; i < nbkps; i++ ) {
    int kpid = vkplmids.at( i ).first;
    int lmid = vkplmids.at( i ).second;

    auto plm = map_manager_->GetMapPoint( lmid );
    if ( plm == nullptr ) {
      continue;
    }
    auto kp = frame.GetKeypointById( kpid );
    if ( kp.lmid_ < 0 ) {
      continue;
    }

    vgoodkpidx.push_back( i );

    vscales.push_back( kp.scale_ );
    vwpts.push_back( plm->GetPoint() );

    vkps.push_back( Eigen::Vector2d( kp.unpx_.x, kp.unpx_.y ) );
    vbvs.push_back( kp.bv_ );
  }

  // If at least 3 correspondances, go
  if ( vkps.size() >= 3 ) {
    bool buse_robust = true;
    bool bapply_l2_after_robust = false;

    // bool success = geometry::tceresMotionOnlyBA(
    //     vkps, vwpts, vscales, Twc, 10, 5.99, buse_robust, bapply_l2_after_robust,
    //     frame.pcalib_leftcam_->fx_, frame.pcalib_leftcam_->fy_, frame.pcalib_leftcam_->cx_,
    //     frame.pcalib_leftcam_->cy_, voutidx );
    bool success = geometry::opencvP3PRansac( vbvs, vwpts, 100, 3., frame.pcalib_leftcam_->fx_,
                                              frame.pcalib_leftcam_->fy_, true, Twc, voutidx );

    for ( const auto& idx : voutidx ) {
      voutlier_idx.push_back( vgoodkpidx.at( idx ) );
    }

    return success;
  }

  return false;
}

void LoopCloser::KNNMatching( const viohw::Frame& newkf, const viohw::Frame& lckf,
                              std::vector<std::pair<int, int>>& vkplmids ) {
  std::vector<int> vkpids, vlmids;
  vkpids.reserve( newkf.nb3dkps_ );
  vlmids.reserve( newkf.nb3dkps_ );

  std::vector<int> vgoodkpids, vgoodlmids;
  vgoodkpids.reserve( newkf.nb3dkps_ );
  vgoodlmids.reserve( newkf.nb3dkps_ );

  cv::Mat query;
  cv::Mat train;

  for ( const auto& kp : newkf.GetKeypoints() ) {
    if ( lckf.isObservingKp( kp.lmid_ ) && kp.is3d_ ) {
      vkplmids.push_back( std::pair<int, int>( kp.lmid_, kp.lmid_ ) );
    } else {
      auto plm = map_manager_->GetMapPoint( kp.lmid_ );
      if ( plm == nullptr ) {
        continue;
      } else if ( !plm->desc_.empty() ) {
        query.push_back( plm->desc_ );
        vkpids.push_back( kp.lmid_ );
      }
    }
  }

  for ( const auto& kp : lckf.GetKeypoints3d() ) {
    if ( newkf.isObservingKp( kp.lmid_ ) ) {
      continue;
    } else {
      auto plm = map_manager_->GetMapPoint( kp.lmid_ );
      if ( plm == nullptr ) {
        continue;
      } else if ( !plm->desc_.empty() ) {
        train.push_back( plm->desc_ );
        vlmids.push_back( kp.lmid_ );
      }
    }
  }

  if ( query.empty() || train.empty() ) {
    return;
  }

  cv::BFMatcher matcher( cv::NORM_HAMMING );
  std::vector<std::vector<cv::DMatch>> vmatches;
  matcher.knnMatch( query, train, vmatches, 2 );

  const int maxdist = query.cols * 0.5 * 8.;

  for ( const auto& m : vmatches ) {
    bool bgood = false;
    if ( m.size() < 2 ) {
      bgood = true;
    } else if ( m.at( 0 ).distance <= maxdist && m.at( 0 ).distance <= m.at( 1 ).distance * 0.85 ) {
      bgood = true;
    }

    if ( bgood ) {
      int kpid = vkpids.at( m.at( 0 ).queryIdx );
      int lmid = vlmids.at( m.at( 0 ).trainIdx );
      vkplmids.push_back( std::pair<int, int>( kpid, lmid ) );
    }
  }

  if ( vkplmids.empty() ) {
    LOG( WARNING ) << "No matches found for LC! Skipping ";
    return;
  }

  LOG( INFO ) << " Found #" << vkplmids.size() << " matches between loop KFs!";
}

std::pair<int, float> LoopCloser::DetectLoop( cv::Mat& cv_descs ) {
  // compute newKF image desc and add to the database
  map_kf_bow_vec_.insert( { new_kf_->kfid_, {} } );
  std::vector<cv::Mat> desc = ConvertToDescriptorVector( cv_descs );
  ORBVocabulary_->transform( desc, map_kf_bow_vec_[new_kf_->kfid_] );

  // matcher to the database, find max score kf
  double max_score = -1e-9;
  int best_matcher_kf_id = -1;
  DBoW2::BowVector cur_kf_vec = map_kf_bow_vec_[new_kf_->kfid_];
  for ( const auto& kf_id_desc : map_kf_bow_vec_ ) {
    int id = kf_id_desc.first;
    DBoW2::BowVector vec = kf_id_desc.second;
    // TODO(new_kf_->kfid_ - id < 40)
    if ( id == new_kf_->kfid_ || new_kf_->kfid_ - id < 40 ) {
      break;
    }
    double similarity_score = ORBVocabulary_->score( cur_kf_vec, vec );
    if ( similarity_score > max_score ) {
      max_score = similarity_score;
      best_matcher_kf_id = id;
    }
  }
  return std::make_pair( best_matcher_kf_id, max_score );
}

void LoopCloser::ComputeDesc( std::vector<cv::KeyPoint>& cv_kps, cv::Mat& cv_descs ) {
  std::vector<Keypoint> kps = new_kf_->GetKeypoints2d();

  cv::Mat mask = cv::Mat( new_kf_img_.rows, new_kf_img_.cols, CV_8UC1, cv::Scalar( 255 ) );
  for ( const auto& kp : kps ) {
    auto plm = map_manager_->GetMapPoint( kp.lmid_ );
    if ( plm == nullptr ) {
      continue;
    } else if ( !plm->desc_.empty() ) {
      cv_kps.push_back( cv::KeyPoint( kp.px_, 5., kp.angle_, 1., kp.scale_ ) );
      cv_descs.push_back( plm->desc_ );
      cv::circle( mask, kp.px_, 2., 0, -1 );
    }
  }
  // brief_cal_->compute( new_kf_img_, cv_kps, cv_descs );

  std::vector<cv::KeyPoint> add_kps;
  fast_detect_->detect( new_kf_img_, add_kps, mask );
  if ( !add_kps.empty() ) {
    cv::KeyPointsFilter::retainBest( add_kps, 300 );

    cv::Mat adddescs;
    brief_cal_->compute( new_kf_img_, add_kps, adddescs );

    if ( !adddescs.empty() ) {
      cv_kps.insert( cv_kps.end(), add_kps.begin(), add_kps.end() );
      cv::vconcat( cv_descs, adddescs, cv_descs );
    }
  }
}

void LoopCloser::AddNewKeyFrame( const FramePtr& pkf, const cv::Mat& im ) {
  std::lock_guard<std::mutex> lock( kf_queen_mutex_ );
  kfs_queen_.push( std::pair<std::shared_ptr<Frame>, cv::Mat>( pkf, im ) );
}

bool LoopCloser::GetNewKeyFrame() {
  std::lock_guard<std::mutex> lock( kf_queen_mutex_ );

  // Check if new KF is available
  if ( kfs_queen_.empty() ) {
    return false;
  }

  // Get most recent KF
  while ( kfs_queen_.size() > 1 ) {
    kfs_queen_.pop();
  }

  new_kf_ = kfs_queen_.front().first;
  new_kf_img_ = kfs_queen_.front().second;
  kfs_queen_.pop();

  return true;
}

std::vector<cv::Mat> LoopCloser::ConvertToDescriptorVector( const cv::Mat& descriptors ) {
  assert( descriptors.rows > 0 );
  std::vector<cv::Mat> desc;
  desc.reserve( descriptors.rows );
  for ( int j = 0; j < descriptors.rows; j++ ) desc.push_back( descriptors.row( j ) );
  return desc;
}

bool LoopCloser::EpipolarFiltering( const Frame& newkf, const Frame& lckf,
                                    vector<std::pair<int, int>>& vkplmids,
                                    vector<int>& voutliers_idx ) {
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  size_t nbkps = vkplmids.size();

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vlcbvs, vcurbvs;
  vlcbvs.reserve( nbkps );
  vcurbvs.reserve( nbkps );
  voutliers_idx.reserve( nbkps );

  for ( const auto& kplmid : vkplmids ) {
    auto kp = newkf.GetKeypointById( kplmid.first );
    vcurbvs.push_back( kp.bv_ );

    auto lckp = lckf.GetKeypointById( kplmid.second );
    vlcbvs.push_back( lckp.bv_ );
  }

  // TODO param config
  bool success = geometry::Opencv5ptEssentialMatrix(
      vlcbvs, vcurbvs, 1000, 3., false, newkf.pcalib_leftcam_->fx_, newkf.pcalib_leftcam_->fy_, R,
      t, voutliers_idx );

  return success;
}

void LoopCloser::RemoveOutliers( std::vector<std::pair<int, int>>& vkplmids,
                                 std::vector<int>& voutliers_idx ) {
  if ( voutliers_idx.empty() ) {
    return;
  }

  size_t nbkps = vkplmids.size();
  std::vector<std::pair<int, int>> vkplmidstmp;

  vkplmidstmp.reserve( nbkps );

  // double pointer
  size_t j = 0;
  for ( size_t i = 0; i < nbkps; i++ ) {
    if ( (int)i != voutliers_idx.at( j ) ) {
      vkplmidstmp.push_back( vkplmids.at( i ) );
    } else {
      j++;
      if ( j == voutliers_idx.size() ) {
        j = 0;
        voutliers_idx.at( 0 ) = -1;
      }
    }
  }

  vkplmids.swap( vkplmidstmp );

  voutliers_idx.clear();
}

}  // namespace viohw
