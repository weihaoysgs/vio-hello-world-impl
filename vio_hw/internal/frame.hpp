#ifndef VIO_HELLO_WORLD_FRAME_HPP
#define VIO_HELLO_WORLD_FRAME_HPP

#include <unordered_set>

#include "sophus/se3.hpp"
#include "vio_hw/internal/camera_calibration.hpp"
#include "vio_hw/internal/keypoint.hpp"

namespace viohw {
class Frame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Frame();
  Frame(std::shared_ptr<CameraCalibration> pcalib_left, const size_t ncellsize);
  Frame(std::shared_ptr<CameraCalibration> pcalib_left);
  Frame(std::shared_ptr<CameraCalibration> pcalib_left,
        std::shared_ptr<CameraCalibration> pcalib_right);
  Frame(std::shared_ptr<CameraCalibration> pcalib_left,
        std::shared_ptr<CameraCalibration> pcalib_right,
        const size_t ncellsize);
  Frame(const Frame &F);

  void updateFrame(const int id, const double img_time);

  std::vector<Keypoint> getKeypoints() const;
  std::vector<Keypoint> getKeypoints2d() const;
  std::vector<Keypoint> getKeypoints3d() const;
  std::vector<Keypoint> getKeypointsStereo() const;

  std::vector<cv::Point2f> getKeypointsPx() const;
  std::vector<cv::Point2f> getKeypointsUnPx() const;
  std::vector<Eigen::Vector3d> getKeypointsBv() const;
  std::vector<int> getKeypointsId() const;
  std::vector<cv::Mat> getKeypointsDesc() const;

  Keypoint getKeypointById(const int lmid) const;

  std::vector<Keypoint> getKeypointsByIds(const std::vector<int> &vlmids) const;

  void computeKeypoint(const cv::Point2f &pt, Keypoint &kp);
  Keypoint computeKeypoint(const cv::Point2f &pt, const int lmid);

  void addKeypoint(const Keypoint &kp);
  void addKeypoint(const cv::Point2f &pt, const int lmid);
  void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc);
  void addKeypoint(const cv::Point2f &pt, const int lmid, const int scale);
  void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc,
                   const int scale);
  void addKeypoint(const cv::Point2f &pt, const int lmid, const cv::Mat &desc,
                   const int scale, const float angle);

  void updateKeypoint(const cv::Point2f &pt, Keypoint &kp);
  void updateKeypoint(const int lmid, const cv::Point2f &pt);
  void updateKeypointDesc(const int lmid, const cv::Mat &desc);
  void updateKeypointAngle(const int lmid, const float angle);

  bool updateKeypointId(const int prevlmid, const int newlmid, const bool is3d);

  void computeStereoKeypoint(const cv::Point2f &pt, Keypoint &kp);
  void updateKeypointStereo(const int lmid, const cv::Point2f &pt);

  void removeKeypoint(const Keypoint &kp);
  void removeKeypointById(const int lmid);

  void removeStereoKeypoint(const Keypoint &kp);
  void removeStereoKeypointById(const int lmid);

  void addKeypointToGrid(const Keypoint &kp);
  void removeKeypointFromGrid(const Keypoint &kp);
  void updateKeypointInGrid(const Keypoint &prevkp, const Keypoint &newkp);
  std::vector<Keypoint> getKeypointsFromGrid(const cv::Point2f &pt) const;
  int getKeypointCellIdx(const cv::Point2f &pt) const;

  std::vector<Keypoint> getSurroundingKeypoints(const Keypoint &kp) const;
  std::vector<Keypoint> getSurroundingKeypoints(const cv::Point2f &pt) const;

  void turnKeypoint3d(const int lmid);

  bool isObservingKp(const int lmid) const;

  Sophus::SE3d getTcw() const;
  Sophus::SE3d getTwc() const;

  Eigen::Matrix3d getRcw() const;
  Eigen::Matrix3d getRwc() const;

  Eigen::Vector3d gettcw() const;
  Eigen::Vector3d gettwc() const;

  void setTwc(const Sophus::SE3d &Twc);
  void setTcw(const Sophus::SE3d &Tcw);

  void setTwc(const Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc);
  void setTcw(const Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw);

  std::set<int> getCovisibleKfSet() const;

  std::map<int, int> getCovisibleKfMap() const;
  void updateCovisibleKfMap(const std::map<int, int> &cokfs);
  void addCovisibleKf(const int kfid);
  void removeCovisibleKf(const int kfid);
  void decreaseCovisibleKf(const int kfid);

  cv::Point2f projCamToImageDist(const Eigen::Vector3d &pt) const;
  cv::Point2f projCamToImage(const Eigen::Vector3d &pt) const;

  cv::Point2f projCamToRightImageDist(const Eigen::Vector3d &pt) const;
  cv::Point2f projCamToRightImage(const Eigen::Vector3d &pt) const;

  cv::Point2f projDistCamToImage(const Eigen::Vector3d &pt) const;
  cv::Point2f projDistCamToRightImage(const Eigen::Vector3d &pt) const;

  Eigen::Vector3d projCamToWorld(const Eigen::Vector3d &pt) const;
  Eigen::Vector3d projWorldToCam(const Eigen::Vector3d &pt) const;

  cv::Point2f projWorldToImage(const Eigen::Vector3d &pt) const;
  cv::Point2f projWorldToImageDist(const Eigen::Vector3d &pt) const;

  cv::Point2f projWorldToRightImage(const Eigen::Vector3d &pt) const;
  cv::Point2f projWorldToRightImageDist(const Eigen::Vector3d &pt) const;

  bool isInImage(const cv::Point2f &pt) const;
  bool isInRightImage(const cv::Point2f &pt) const;

  void displayFrameInfo();

  // For using frame in ordered containers
  bool operator<(const Frame &f) const { return id_ < f.id_; }

  void reset();

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
