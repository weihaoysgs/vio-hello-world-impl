#ifndef VIO_HELLO_WORLD_MAP_MANAGER_HPP
#define VIO_HELLO_WORLD_MAP_MANAGER_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "vio_hw/internal/feat/feature.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_point.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"

namespace viohw {

class MapManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // constructor function
  MapManager( SettingPtr state, FramePtr pframe, FeatureBasePtr pfeatextract,
              TrackerBasePtr ptracker );

  // de constructor function
  ~MapManager() = default;

  // create a new keyframe from image raw
  void CreateKeyframe( const cv::Mat &im, const cv::Mat &im_raw );

  // prepare frame, traverse frame keypoint, add keyframe observations for each keypoint
  void PrepareFrame();

  // insert keyframe to [map_kfs_], [num_kfs_++ and kf_id_++]
  void AddKeyframe();

  // extract keypoint and update desc for every keypoint
  void ExtractKeypoints( const cv::Mat &im, const cv::Mat &im_raw );

  // stereo matching for left and right image
  void StereoMatching( Frame &frame, const std::vector<cv::Mat> &vleftpyr,
                       const std::vector<cv::Mat> &vrightpyr );

  // add keypoint to [map_lms_], num_lms_++ and lm_id_++
  void AddMapPoint();

  // add keypoint(with desc) to [map_lms_], num_lms_++ and lm_id_++
  void AddMapPoint( const cv::Mat &desc );

  // add keypoint to frame and [map_lms_]
  void AddKeypointsToFrame( const std::vector<cv::Point2f> &vpts, Frame &frame );

  // add keypoint(with desc) to frame and [map_lms_]
  void AddKeypointsToFrame( const std::vector<cv::Point2f> &vpts,
                            const std::vector<cv::Mat> &vdescs, Frame &frame );

  // remove keypoint from current frame
  void RemoveObsFromCurFrameById( const int lmid );

  // get keyframe from id
  std::shared_ptr<Frame> GetKeyframe( const int kfid ) const;

  // get mappoint by id
  std::shared_ptr<MapPoint> GetMapPoint( const int lmid ) const;

  // update mappoint
  void UpdateMapPoint( const int lmid, const Eigen::Vector3d &wpt,
                       const double kfanch_invdepth = -1 );

  // remove mappoint obs in keyframe
  void RemoveMapPointObs( const int lmid, const int kfid );

  void RemoveMapPoint(const int lmid);

  // compute and update keypoint desc
  void DescribeKeypoints( const cv::Mat &im, const std::vector<Keypoint> &vkps,
                          const std::vector<cv::Point2f> &vpts );

  // get number of keyframe
  int GetNumberKF() const;

  void NumKFPlus();

  int GetNumberLandmark() const;

  void NumLandmarkPlus();

  FramePtr GetCurrentFrame() { return current_frame_; }

public:
  mutable std::mutex kf_mutex_, lm_mutex_;
  mutable std::mutex map_mutex_, optim_mutex_;
  mutable std::mutex num_kfs_mutex_, num_lms_mutex_;

private:
  int lm_id_, kf_id_;
  int num_lms_, num_kfs_;

  SettingPtr param_;
  FramePtr current_frame_;
  FeatureBasePtr feature_extractor_;
  TrackerBasePtr tracker_;

  std::unordered_map<int, std::shared_ptr<Frame>> map_kfs_;
  std::unordered_map<int, std::shared_ptr<MapPoint>> map_lms_;
};

typedef std::shared_ptr<MapManager> MapManagerPtr;
typedef std::shared_ptr<const MapManager> MapManagerConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_MAP_MANAGER_HPP
