#include "vio_hw/internal/frame.hpp"

namespace viohw {

void Frame::updateFrame(const int id, const double img_time) {
  id_ = id;
  img_time_ = img_time;
}

Frame::Frame(std::shared_ptr<CameraCalibration> calib_left,
             std::shared_ptr<CameraCalibration> calib_right, const size_t ncellsize)
    : id_(-1),
      kfid_(0),
      img_time_(0.),
      ncellsize_(ncellsize),
      nbkps_(0),
      nb2dkps_(0),
      nb3dkps_(0),
      nb_stereo_kps_(0),
      pcalib_leftcam_(calib_left),
      pcalib_rightcam_(calib_right) {
  Eigen::Vector3d t = pcalib_rightcam_->Tcic0_.translation();
  Eigen::Matrix3d tskew;
  tskew << 0., -t(2), t(1), t(2), 0., -t(0), -t(1), t(0), 0.;

  Eigen::Matrix3d R = pcalib_rightcam_->Tcic0_.rotationMatrix();

  Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R * pcalib_leftcam_->iK_;

  cv::eigen2cv(Frl_, Fcv_);

  // Init grid from images size
  nbwcells_ = ceil((float)pcalib_leftcam_->img_w_ / ncellsize_);
  nbhcells_ = ceil((float)pcalib_leftcam_->img_h_ / ncellsize_);
  ngridcells_ = nbwcells_ * nbhcells_;
  noccupcells_ = 0;

  vgridkps_.resize(ngridcells_);
}

Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize)
    : id_(-1),
      kfid_(0),
      img_time_(0.),
      ncellsize_(ncellsize),
      nbkps_(0),
      nb2dkps_(0),
      nb3dkps_(0),
      nb_stereo_kps_(0),
      pcalib_leftcam_(pcalib_left) {
  // Init grid from images size
  nbwcells_ = static_cast<size_t>(ceilf(static_cast<float>(pcalib_leftcam_->img_w_) / ncellsize_));
  nbhcells_ = static_cast<size_t>(ceilf(static_cast<float>(pcalib_leftcam_->img_h_) / ncellsize_));
  ngridcells_ = nbwcells_ * nbhcells_;
  noccupcells_ = 0;

  vgridkps_.resize(ngridcells_);
}

Frame::Frame(const Frame &F)
    : id_(F.id_),
      kfid_(F.kfid_),
      img_time_(F.img_time_),
      mapkps_(F.mapkps_),
      vgridkps_(F.vgridkps_),
      ngridcells_(F.ngridcells_),
      noccupcells_(F.noccupcells_),
      ncellsize_(F.ncellsize_),
      nbwcells_(F.nbwcells_),
      nbhcells_(F.nbhcells_),
      nbkps_(F.nbkps_),
      nb2dkps_(F.nb2dkps_),
      nb3dkps_(F.nb3dkps_),
      nb_stereo_kps_(F.nb_stereo_kps_),
      Twc_(F.Twc_),
      Tcw_(F.Tcw_),
      pcalib_leftcam_(F.pcalib_leftcam_),
      pcalib_rightcam_(F.pcalib_rightcam_),
      Frl_(F.Frl_),
      Fcv_(F.Fcv_),
      map_covkfs_(F.map_covkfs_),
      set_local_mapids_(F.set_local_mapids_) {}

std::vector<Keypoint> Frame::getKeypoints() const {
  std::lock_guard<std::mutex> lock(kps_mutex_);

  std::vector<Keypoint> v;
  v.reserve(nbkps_);
  for (const auto &kp : mapkps_) {
    v.push_back(kp.second);
  }
  return v;
}
}  // namespace viohw
