#include "vio_hw/internal/estimator.hpp"

namespace viohw {

Estimator::Estimator( SettingPtr param, MapManagerPtr map_manager, OptimizationPtr optimization )
    : param_( param ), map_manager_( map_manager ), optimization_( optimization ) {}

void Estimator::run() {
  bool open_backend = param_->backend_optimization_setting_.open_backend_opt_;
  if ( !open_backend ) {
    LOG( WARNING ) << "<<<Backend Optimization is close>>>";
    return;
  }

  while ( true ) {
    if ( GetNewKf() ) {
      ApplyLocalBA();
    } else {
      std::chrono::microseconds dura( 20 );
      std::this_thread::sleep_for( dura );
    }
  }
}

void Estimator::ApplyLocalBA() {
  int mincstkfs = 2;

  if ( newkf_->kfid_ < mincstkfs ) {
    return;
  }
  if ( newkf_->nb3dkps_ == 0 ) {
    return;
  }

  std::lock_guard<std::mutex> lock2( map_manager_->optim_mutex_ );

  // TODO
  // We signal that Estimator is performing BA
  // pslamstate_->blocalba_is_on_ = true;

  bool use_robust_cost = true;
  optimization_->LocalBA( *newkf_, use_robust_cost );

  // We signal that Estimator is stopping BA
  // pslamstate_->blocalba_is_on_ = false;
}

bool Estimator::GetNewKf() {
  std::lock_guard<std::mutex> lock( qkf_mutex_ );

  // Check if new KF is available
  if ( queen_kfs_.empty() ) {
    is_newkf_available_ = false;
    return false;
  }

  // In SLAM-mode, we only processed the last received KF
  // but we trick the covscore if several KFs were waiting
  // to make sure that they are all optimized
  std::vector<int> vkfids;
  vkfids.reserve( queen_kfs_.size() );
  while ( queen_kfs_.size() > 1 ) {
    queen_kfs_.pop();
    vkfids.push_back( newkf_->kfid_ );
  }
  newkf_ = queen_kfs_.front();
  queen_kfs_.pop();

  if ( !vkfids.empty() ) {
    // TODO
    // for( const auto &kfid : vkfids ) {
    //   pnewkf_->map_covkfs_[kfid] = pnewkf_->nb3dkps_;
    // }

    LOG( WARNING ) << "ESTIMATOR is late! Skip " << vkfids.size() << " KF";
  }
  is_newkf_available_ = false;

  return true;
}

void Estimator::AddNewKf( const std::shared_ptr<Frame>& kf ) {
  std::lock_guard<std::mutex> lock( qkf_mutex_ );
  queen_kfs_.push( kf );
  is_newkf_available_ = true;

  // TODO
  // We signal that a new KF is ready
  // if( pslamstate_->blocalba_is_on_
  //      && !poptimizer_->stopLocalBA() )
  // {
  //   poptimizer_->signalStopLocalBA();
  // }
}
}  // namespace viohw