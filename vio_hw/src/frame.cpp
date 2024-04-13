#include "vio_hw/internal/frame.hpp"

namespace viohw {

void Frame::updateFrame( const int id, const double img_time ) {
  id_ = id;
  img_time_ = img_time;
}

Frame::Frame( std::shared_ptr<CameraCalibration> calib_left,
              std::shared_ptr<CameraCalibration> calib_right, const size_t ncellsize )
    : id_( -1 ),
      kfid_( 0 ),
      img_time_( 0. ),
      ncellsize_( ncellsize ),
      nbkps_( 0 ),
      nb2dkps_( 0 ),
      nb3dkps_( 0 ),
      nb_stereo_kps_( 0 ),
      pcalib_leftcam_( calib_left ),
      pcalib_rightcam_( calib_right ) {
  Eigen::Vector3d t = pcalib_rightcam_->Tcic0_.translation();
  Eigen::Matrix3d tskew;
  tskew << 0., -t( 2 ), t( 1 ), t( 2 ), 0., -t( 0 ), -t( 1 ), t( 0 ), 0.;

  Eigen::Matrix3d R = pcalib_rightcam_->Tcic0_.rotationMatrix();

  Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R * pcalib_leftcam_->iK_;

  cv::eigen2cv( Frl_, Fcv_ );

  // Init grid from images size
  nbwcells_ = ceil( (float)pcalib_leftcam_->img_w_ / ncellsize_ );
  nbhcells_ = ceil( (float)pcalib_leftcam_->img_h_ / ncellsize_ );
  ngridcells_ = nbwcells_ * nbhcells_;
  noccupcells_ = 0;

  vgridkps_.resize( ngridcells_ );
}

Frame::Frame( std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize )
    : id_( -1 ),
      kfid_( 0 ),
      img_time_( 0. ),
      ncellsize_( ncellsize ),
      nbkps_( 0 ),
      nb2dkps_( 0 ),
      nb3dkps_( 0 ),
      nb_stereo_kps_( 0 ),
      pcalib_leftcam_( pcalib_left ) {
  // Init grid from images size
  nbwcells_ =
      static_cast<size_t>( ceilf( static_cast<float>( pcalib_leftcam_->img_w_ ) / ncellsize_ ) );
  nbhcells_ =
      static_cast<size_t>( ceilf( static_cast<float>( pcalib_leftcam_->img_h_ ) / ncellsize_ ) );
  ngridcells_ = nbwcells_ * nbhcells_;
  noccupcells_ = 0;

  vgridkps_.resize( ngridcells_ );
}

Frame::Frame( const Frame &F )
    : id_( F.id_ ),
      kfid_( F.kfid_ ),
      img_time_( F.img_time_ ),
      mapkps_( F.mapkps_ ),
      vgridkps_( F.vgridkps_ ),
      ngridcells_( F.ngridcells_ ),
      noccupcells_( F.noccupcells_ ),
      ncellsize_( F.ncellsize_ ),
      nbwcells_( F.nbwcells_ ),
      nbhcells_( F.nbhcells_ ),
      nbkps_( F.nbkps_ ),
      nb2dkps_( F.nb2dkps_ ),
      nb3dkps_( F.nb3dkps_ ),
      nb_stereo_kps_( F.nb_stereo_kps_ ),
      Twc_( F.Twc_ ),
      Tcw_( F.Tcw_ ),
      pcalib_leftcam_( F.pcalib_leftcam_ ),
      pcalib_rightcam_( F.pcalib_rightcam_ ),
      Frl_( F.Frl_ ),
      Fcv_( F.Fcv_ ),
      map_covkfs_( F.map_covkfs_ ),
      set_local_mapids_( F.set_local_mapids_ ) {
  // TODO reset preintegrated measure
}

void Frame::AddKeypoint( const Keypoint &kp ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  if ( mapkps_.count( kp.lmid_ ) ) {
    LOG( WARNING )
        << "WEIRD!  Trying to add a KP with an already existing lmid... Not gonna do it!";
    return;
  }

  mapkps_.emplace( kp.lmid_, kp );
  // TODO
  // addKeypointToGrid(kp);

  nbkps_++;
  if ( kp.is3d_ ) {
    nb3dkps_++;
  } else {
    nb2dkps_++;
  }
}

void Frame::AddKeypoint( const cv::Point2f &pt, const int lmid ) {
  Keypoint kp = ComputeKeypoint( pt, lmid );

  AddKeypoint( kp );
}

void Frame::AddKeypoint( const cv::Point2f &pt, const int lmid, const cv::Mat &desc ) {
  Keypoint kp = ComputeKeypoint( pt, lmid );
  kp.desc_ = desc;

  AddKeypoint( kp );
}

Keypoint Frame::ComputeKeypoint( const cv::Point2f &pt, const int lmid ) {
  Keypoint kp;
  kp.lmid_ = lmid;
  this->ComputeKeypoint( pt, kp );
  return kp;
}

void Frame::ComputeKeypoint( const cv::Point2f &pt, Keypoint &kp ) {
  kp.px_ = pt;
  kp.unpx_ = pcalib_leftcam_->undistortImagePoint( pt );

  Eigen::Vector3d hunpx( kp.unpx_.x, kp.unpx_.y, 1. );
  kp.bv_ = pcalib_leftcam_->iK_ * hunpx;
  kp.bv_.normalize();
}

void Frame::UpdateKeypoint( const int lmid, const cv::Point2f &pt ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return;
  }

  Keypoint upkp = it->second;

  if ( upkp.is_stereo_ ) {
    nb_stereo_kps_--;
    upkp.is_stereo_ = false;
  }

  ComputeKeypoint( pt, upkp );

  // TODO
  // updateKeypointInGrid(it->second, upkp);
  it->second = upkp;
}

void Frame::RemoveKeypointById( const int lmid ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return;
  }

  // TODO
  // removeKeypointFromGrid(it->second);

  if ( it->second.is3d_ ) {
    nb3dkps_--;
  } else {
    nb2dkps_--;
  }
  nbkps_--;
  if ( it->second.is_stereo_ ) {
    nb_stereo_kps_--;
  }
  mapkps_.erase( lmid );
}

Keypoint Frame::GetKeypointById( const int lmid ) const {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return Keypoint();
  }

  return it->second;
}

void Frame::UpdateKeypointStereo( const int lmid, const cv::Point2f &pt ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return;
  }

  ComputeStereoKeypoint( pt, it->second );
}

bool Frame::isObservingKp( const int lmid ) const {
  std::lock_guard<std::mutex> lock( kps_mutex_ );
  return mapkps_.count( lmid );
}

// Compute stereo keypoint from raw pixel position
void Frame::ComputeStereoKeypoint( const cv::Point2f &pt, Keypoint &kp ) {
  kp.rpx_ = pt;
  kp.runpx_ = pcalib_rightcam_->undistortImagePoint( pt );

  Eigen::Vector3d bv( kp.runpx_.x, kp.runpx_.y, 1. );
  bv = pcalib_rightcam_->iK_ * bv.eval();
  bv.normalize();

  kp.rbv_ = bv;

  if ( !kp.is_stereo_ ) {
    kp.is_stereo_ = true;
    nb_stereo_kps_++;
  }
}
// Return vector of stereo keypoint objects
std::vector<Keypoint> Frame::getKeypointsStereo() const {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  std::vector<Keypoint> v;
  v.reserve( nb_stereo_kps_ );
  for ( const auto &kp : mapkps_ ) {
    if ( kp.second.is_stereo_ ) {
      v.push_back( kp.second );
    }
  }
  return v;
}
// Return vector of 2D keypoint objects
std::vector<Keypoint> Frame::GetKeypoints2d() const {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  std::vector<Keypoint> v;
  v.reserve( nb2dkps_ );
  for ( const auto &kp : mapkps_ ) {
    if ( !kp.second.is3d_ ) {
      v.push_back( kp.second );
    }
  }
  return v;
}

void Frame::UpdateKeypointDesc( const int lmid, const cv::Mat &desc ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return;
  }

  it->second.desc_ = desc;
}

// Return vector of 3D keypoint objects
std::vector<Keypoint> Frame::GetKeypoints3d() const {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  std::vector<Keypoint> v;
  v.reserve( nb3dkps_ );
  for ( const auto &kp : mapkps_ ) {
    if ( kp.second.is3d_ ) {
      v.push_back( kp.second );
    }
  }
  return v;
}

// Return vector of keypoint objects
std::vector<Keypoint> Frame::GetKeypoints() const {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  std::vector<Keypoint> v;
  v.reserve( nbkps_ );
  for ( const auto &kp : mapkps_ ) {
    v.push_back( kp.second );
  }
  return v;
}

void Frame::RemoveStereoKeypointById( const int lmid ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return;
  }

  if ( it->second.is_stereo_ ) {
    it->second.is_stereo_ = false;
    nb_stereo_kps_--;
  }
}

void Frame::TurnKeypoint3d( const int lmid ) {
  std::lock_guard<std::mutex> lock( kps_mutex_ );

  auto it = mapkps_.find( lmid );
  if ( it == mapkps_.end() ) {
    return;
  }

  if ( !it->second.is3d_ ) {
    it->second.is3d_ = true;
    nb3dkps_++;
    nb2dkps_--;
  }
}

cv::Point2f Frame::ProjCamToImage( const Eigen::Vector3d &pt ) const {
  return pcalib_leftcam_->projectCamToImage( pt );
}

cv::Point2f Frame::ProjCamToRightImage( const Eigen::Vector3d &pt ) const {
  return pcalib_rightcam_->projectCamToImage( pcalib_rightcam_->Tcic0_ * pt );
}

Eigen::Vector3d Frame::ProjWorldToCam( const Eigen::Vector3d &pt ) const {
  std::lock_guard<std::mutex> lock( pose_mutex_ );

  Eigen::Vector3d campt = Tcw_ * pt;

  return campt;
}
Eigen::Vector3d Frame::ProjCamToWorld( const Eigen::Vector3d &pt ) const {
  std::lock_guard<std::mutex> lock( pose_mutex_ );

  Eigen::Vector3d wpt = Twc_ * pt;

  return wpt;
}
Sophus::SE3d Frame::GetTwc() const {
  std::lock_guard<std::mutex> lock( pose_mutex_ );
  return Twc_;
}
Sophus::SE3d Frame::GetTcw() const {
  std::lock_guard<std::mutex> lock( pose_mutex_ );
  return Tcw_;
}
void Frame::SetTwc( const Sophus::SE3d &Twc ) {
  std::lock_guard<std::mutex> lock( pose_mutex_ );

  Twc_ = Twc;
  Tcw_ = Twc.inverse();
}
}  // namespace viohw
