#include "vio_hw/internal//map_point.hpp"

namespace viohw {
MapPoint::MapPoint( const int lmid, const int kfid, const bool bobs )
    : lmid_( lmid ), isobs_( bobs ), kfid_( kfid ), invdepth_( -1. ) {
  set_kfids_.insert( kfid );
  is3d_ = false;
  ptxyz_.setZero();
}

MapPoint::MapPoint( const int lmid, const int kfid, const cv::Mat &desc, const bool bobs )
    : lmid_( lmid ), isobs_( bobs ), kfid_( kfid ), invdepth_( -1. ) {
  set_kfids_.insert( kfid );

  map_kf_desc_.emplace( kfid, desc );
  map_desc_dist_.emplace( kfid, 0. );
  desc_ = map_kf_desc_.at( kfid );

  is3d_ = false;
  ptxyz_.setZero();
}

std::set<int> MapPoint::GetKfObsSet() const {
  std::lock_guard<std::mutex> lock( pt_mutex );
  return set_kfids_;
}
void MapPoint::RemoveKfObs( const int kfid ) {
  std::lock_guard<std::mutex> lock( pt_mutex );

  if ( !set_kfids_.count( kfid ) ) {
    return;
  }

  // First remove the related id
  set_kfids_.erase( kfid );

  if ( set_kfids_.empty() ) {
    desc_.release();
    map_kf_desc_.clear();
    map_desc_dist_.clear();
    return;
  }

  // Set new KF anchor if removed
  if ( kfid == kfid_ ) {
    kfid_ = *set_kfids_.begin();
  }

  // Then figure out the most representative one
  // (we could also use the last one to save time)
  float mindist = desc_.cols * 8.;
  int minid = -1;

  auto itdesc = map_kf_desc_.find( kfid );
  if ( itdesc != map_kf_desc_.end() ) {
    for ( const auto &kf_d : map_kf_desc_ ) {
      if ( kf_d.first != kfid ) {
        float dist = cv::norm( itdesc->second, kf_d.second, cv::NORM_HAMMING );
        float &descdist = map_desc_dist_.find( kf_d.first )->second;
        descdist -= dist;

        // Get the lowest one
        if ( descdist < mindist ) {
          mindist = descdist;
          minid = kf_d.first;
        }
      }
    }

    itdesc->second.release();
    map_kf_desc_.erase( kfid );
    map_desc_dist_.erase( kfid );

    // Remove desc / update mean desc
    if ( minid > 0 ) {
      desc_ = map_kf_desc_.at( minid );
    }
  }
}
void MapPoint::AddDesc( const int kfid, const cv::Mat &d ) {
  std::lock_guard<std::mutex> lock( pt_mutex );

  auto it = map_kf_desc_.find( kfid );
  if ( it != map_kf_desc_.end() ) {
    return;
  }

  // First add the desc and init its distance score
  map_kf_desc_.emplace( kfid, d );

  map_desc_dist_.emplace( kfid, 0 );
  float &newdescdist = map_desc_dist_.find( kfid )->second;

  if ( map_kf_desc_.size() == 1 ) {
    desc_ = d;
    return;
  }

  // Then figure out the most representative one
  // (we could also use the last one to save time)
  float mindist = desc_.cols * 8.;
  int minid = -1;

  // Then update the distance scores for all desc
  for ( const auto &kf_d : map_kf_desc_ ) {
    float dist = cv::norm( d, kf_d.second, cv::NORM_HAMMING );

    // Update previous desc
    map_desc_dist_.at( kf_d.first ) += dist;

    // Get the lowest one
    if ( dist < mindist ) {
      mindist = dist;
      minid = kf_d.first;
    }

    // Update new desc
    newdescdist += dist;
  }

  // Get the lowest one
  if ( newdescdist < mindist ) {
    minid = kfid;
  }

  desc_ = map_kf_desc_.at( minid );
}

Eigen::Vector3d MapPoint::GetPoint() const {
  std::lock_guard<std::mutex> lock( pt_mutex );
  return ptxyz_;
}

void MapPoint::SetPoint( const Eigen::Vector3d &ptxyz, const double kfanch_invdepth ) {
  std::lock_guard<std::mutex> lock( pt_mutex );
  ptxyz_ = ptxyz;
  is3d_ = true;
  if ( kfanch_invdepth >= 0. ) {
    invdepth_ = kfanch_invdepth;
  }
}
}  // namespace viohw