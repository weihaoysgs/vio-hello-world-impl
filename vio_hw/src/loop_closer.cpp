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
  map_kf_bow_vec_.insert( { new_kf_->kfid_, {} } );
  std::vector<cv::Mat> desc = ConvertToDescriptorVector( cv_descs );
  ORBVocabulary_->transform( desc, map_kf_bow_vec_[new_kf_->kfid_] );
  double max_score = -1e-9;
  int best_matcher_kf_id = -1;
  DBoW2::BowVector cur_kf_vec = map_kf_bow_vec_[new_kf_->kfid_];
  for ( const auto& kf_id_desc : map_kf_bow_vec_ ) {
    int id = kf_id_desc.first;
    DBoW2::BowVector vec = kf_id_desc.second;
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
    } else {
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

}  // namespace viohw
