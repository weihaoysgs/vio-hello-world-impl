#ifndef VIO_HELLO_WORLD_FRAME_HPP
#define VIO_HELLO_WORLD_FRAME_HPP

#include <glog/logging.h>

#include <unordered_set>

#include "sophus/se3.hpp"
#include "vio_hw/internal/camera_calibration.hpp"
#include "vio_hw/internal/keypoint.hpp"

namespace viohw {
class Frame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Frame() = default;
  Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize);

  Frame(std::shared_ptr<CameraCalibration> pcalib_left,
        std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize);
  Frame(const Frame &F);

  void updateFrame(const int id, const double img_time);

  std::vector<Keypoint> getKeypoints() const;
  void AddKeypoint(const Keypoint &kp);
  void AddKeypoint(const cv::Point2f &pt, const int lmid);
  void AddKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc);
  Keypoint ComputeKeypoint(const cv::Point2f &pt, const int lmid);
  void ComputeKeypoint(const cv::Point2f &pt, Keypoint &kp);
  void UpdateKeypoint(const int lmid, const cv::Point2f &pt);
  void RemoveKeypointById(const int lmid);
  Keypoint GetKeypointById(const int lmid) const;
  void ComputeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp);
  std::vector<Keypoint> getKeypointsStereo() const;
  void RemoveStereoKeypointById(const int lmid);
  std::vector<Keypoint> GetKeypoints2d() const;
  void UpdateKeypointStereo(const int lmid, const cv::Point2f &pt);
  cv::Point2f ProjCamToRightImage(const Eigen::Vector3d &pt) const;
  Eigen::Vector3d ProjCamToWorld(const Eigen::Vector3d &pt) const;
  Sophus::SE3d GetTwc() const;
  Sophus::SE3d GetTcw() const;
  void SetTwc(const Sophus::SE3d &Twc);
  cv::Point2f ProjCamToImage(const Eigen::Vector3d &pt) const;
  void TurnKeypoint3d(const int lmid);
  // For using frame in ordered containers
  bool operator<(const Frame &f) const { return id_ < f.id_; }

  // Frame info
  int id_, kfid_;
  double img_time_;

  // Hash Map of observed keypoints
  std::unordered_map<int, Keypoint> mapkps_;

  // Grid of kps sorted by cell numbers and scale
  // (We use const pointer to reference the keypoints in vkps_
  // HENCE we should only use the grid to read kps)
  std::vector<std::vector<int>> vgridkps_;
  size_t ngridcells_, noccupcells_, ncellsize_, nbwcells_, nbhcells_;

  size_t nbkps_, nb2dkps_, nb3dkps_, nb_stereo_kps_;

  // Pose (T cam -> world), (T world -> cam)
  Sophus::SE3d Twc_, Tcw_;
  Sophus::SE3d Twb_, Tbw_;
  /* TODO
  Set a vector of calib ptrs to handle any multicam system.
  Each calib ptr should contain an extrinsic parametrization with a common
  reference frame. If cam0 is the ref., its extrinsic would be the identity.
  Would mean an easy integration of IMU body frame as well.
  */
  // Calibration model
  std::shared_ptr<CameraCalibration> pcalib_leftcam_;
  std::shared_ptr<CameraCalibration> pcalib_rightcam_;

  Eigen::Matrix3d Frl_;
  cv::Mat Fcv_;

  // Covisible kf ids
  std::map<int, int> map_covkfs_;

  // Local MapPoint ids
  std::unordered_set<int> set_local_mapids_;

  // Mutex
  mutable std::mutex kps_mutex_, pose_mutex_;
  mutable std::mutex grid_mutex_, cokfs_mutex_;
};

typedef std::shared_ptr<Frame> FramePtr;
typedef std::shared_ptr<const Frame> FrameConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_FRAME_HPP
