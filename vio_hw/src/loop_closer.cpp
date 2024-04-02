#include "vio_hw/internal/loop_closer.hpp"

namespace viohw {

LoopCloser::LoopCloser( SettingPtr param, MapManagerPtr map_manager )
    : param_( param ), map_manager_( map_manager ) {
  use_loop_ = param->loop_setting_.use_loop_closer_;
  loop_threshold_ = param->loop_setting_.loop_threshold_;

  if ( use_loop_ ) {
    ORBVocabulary_ = std::make_unique<ORBVocabulary>();
    fast_detect_ = cv::FastFeatureDetector::create( 20 );
    brief_cal_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
  } else {
    LOG( WARNING ) << ">>>>Loop Closing is close.>>>>";
  }
}

void LoopCloser::run() {
  while ( true ) {
    if ( GetNewKeyFrame() ) {
      LOG( INFO ) << "Loop thread get new kf";
    } else {
      std::chrono::microseconds dura( 100 );
      std::this_thread::sleep_for( dura );
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
}  // namespace viohw
