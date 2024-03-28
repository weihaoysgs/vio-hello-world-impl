#include "vio_hw/internal/frame.hpp"

namespace viohw {

void Frame::updateFrame(const int id, const double img_time) {
  id_ = id;
  img_time_ = img_time;
}

Frame::Frame(std::shared_ptr<CameraCalibration> calib_left,
             std::shared_ptr<CameraCalibration> calib_right,
             const size_t ncellsize)
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

  Frl_ = pcalib_rightcam_->K_.transpose().inverse() * tskew * R *
         pcalib_leftcam_->iK_;

  cv::eigen2cv(Frl_, Fcv_);

  // Init grid from images size
  nbwcells_ = ceil((float)pcalib_leftcam_->img_w_ / ncellsize_);
  nbhcells_ = ceil((float)pcalib_leftcam_->img_h_ / ncellsize_);
  ngridcells_ = nbwcells_ * nbhcells_;
  noccupcells_ = 0;

  vgridkps_.resize(ngridcells_);
}

Frame::Frame(std::shared_ptr<CameraCalibration> pcalib_left,
             const size_t ncellsize)
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
  nbwcells_ = static_cast<size_t>(
      ceilf(static_cast<float>(pcalib_leftcam_->img_w_) / ncellsize_));
  nbhcells_ = static_cast<size_t>(
      ceilf(static_cast<float>(pcalib_leftcam_->img_h_) / ncellsize_));
  ngridcells_ = nbwcells_ * nbhcells_;
  noccupcells_ = 0;

  vgridkps_.resize(ngridcells_);
}
}  // namespace viohw
