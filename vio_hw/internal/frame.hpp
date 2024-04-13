#ifndef VIO_HELLO_WORLD_FRAME_HPP
#define VIO_HELLO_WORLD_FRAME_HPP

#include <glog/logging.h>

#include <unordered_set>

#include "backend/sensor_fusion/imu/imu_frame_state.h"
#include "sophus/se3.hpp"
#include "vio_hw/internal/camera_calibration.hpp"
#include "vio_hw/internal/keypoint.hpp"

namespace viohw {
class Frame
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Frame() = default;

  // constructor for mono image
  Frame( std::shared_ptr<CameraCalibration> calib_left, const size_t cell_size );

  // constructor for stereo image
  Frame( std::shared_ptr<CameraCalibration> pcalib_left,
         std::shared_ptr<CameraCalibration> pcalib_right, const size_t ncellsize );

  // copy constructor
  Frame( const Frame &F );

  // update frame id and time
  void updateFrame( const int id, const double img_time );

  // add keypoint to current frame, insert to [mapkps_], and plus num of [2d/3d/total] kps
  void AddKeypoint( const Keypoint &kp );

  // compute the keypoint attributes and add to the current frame
  void AddKeypoint( const cv::Point2f &pt, const int lmid );

  // function overload, compute the keypoint with desc and add to the current frame
  void AddKeypoint( const cv::Point2f &pt, const int lmid, const cv::Mat &desc );

  // function Overload, compute keypoint with specified [lmid]
  Keypoint ComputeKeypoint( const cv::Point2f &pt, const int lmid );

  // compute keypoint according to the cv::Point, compute undistort image coordinate and normalized
  // plane coordinate
  void ComputeKeypoint( const cv::Point2f &pt, Keypoint &kp );

  // update the keypoint in current frame, if [stereo] set to [false]
  void UpdateKeypoint( const int lmid, const cv::Point2f &pt );

  // remove keypoint by id, minus number of [stereo/3d/2d/total]
  void RemoveKeypointById( const int lmid );

  // get keypoint by lmid
  Keypoint GetKeypointById( const int lmid ) const;

  // compute stereo keypoint attributes and plus [nb_stereo_kps_]
  void ComputeStereoKeypoint( const cv::Point2f &pt, Keypoint &kp );

  // get all stereo keypoint
  std::vector<Keypoint> getKeypointsStereo() const;

  // remove stereo keypoint by lmid, erase from [mapkps_], minus [nb_stereo_kps_]
  void RemoveStereoKeypointById( const int lmid );

  // the keypoint with id[lmid] is observed by current frame
  bool isObservingKp( const int lmid ) const;

  // get all 2D keypoint
  std::vector<Keypoint> GetKeypoints2d() const;

  // get all 3d keypoint
  std::vector<Keypoint> GetKeypoints3d() const;

  // update keypoint desc
  void UpdateKeypointDesc( const int lmid, const cv::Mat &desc );

  // get all keypoints
  std::vector<Keypoint> GetKeypoints() const;

  // update stereo keypoint, recompute the keypoint attributes
  void UpdateKeypointStereo( const int lmid, const cv::Point2f &pt );

  // projection 3d point from camera to right image pixel coordinate
  cv::Point2f ProjCamToRightImage( const Eigen::Vector3d &pt ) const;

  // projection 3d point from world to camera
  Eigen::Vector3d ProjWorldToCam( const Eigen::Vector3d &pt ) const;

  // projection 3d point from camera to world
  Eigen::Vector3d ProjCamToWorld( const Eigen::Vector3d &pt ) const;

  // projection 3d point from camera to left image pixel coordinate
  cv::Point2f ProjCamToImage( const Eigen::Vector3d &pt ) const;

  // get Twc
  Sophus::SE3d GetTwc() const;

  // get Tcw
  Sophus::SE3d GetTcw() const;

  // set Twc
  void SetTwc( const Sophus::SE3d &Twc );

  // make keypoint is 3D, update related attributes, number of [2D/3D] kps, [is_3d_]
  void TurnKeypoint3d( const int lmid );

  // For using frame in ordered containers
  bool operator<( const Frame &f ) const { return id_ < f.id_; }

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

  // imu state
  backend::IMU::IMUState imu_state_;

};

typedef std::shared_ptr<Frame> FramePtr;
typedef std::shared_ptr<const Frame> FrameConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_FRAME_HPP
