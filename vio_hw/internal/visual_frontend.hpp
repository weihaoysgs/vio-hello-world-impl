#ifndef VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP
#define VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP

#include "vio_hw/internal/constant_motion_model.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"
#include "vio_hw/internal/viz/visualization_base.hpp"
#include "geometry/motion_ba/motion_ba.hpp"

namespace viohw {
class VisualFrontEnd
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // default construction function
  VisualFrontEnd() = default;

  // default deconstruction function
  ~VisualFrontEnd() = default;

  // construction function
  VisualFrontEnd(SettingPtr setting, FramePtr frame, MapManagerPtr manager,
                 TrackerBasePtr tracker, VisualizationBasePtr viz);

  // tracking left image
  bool TrackerMono(cv::Mat &image, double time);

  // tracking image and build keyframe
  bool VisualTracking(cv::Mat &image, double time);

  // preprocess image, build pyramid for tracking and apply clahe
  void PreProcessImage(cv::Mat &image);

  // optical flow tracking for frame before and after
  void KLTTracking();

  // filter outlier via epipolar constraint
  void Epipolar2d2dFiltering();

  // draw tracker result and show in ui
  void ShowTrackingResult();

  // check current frame is keyframe
  bool CheckIsNewKeyframe();

  // compute visual frontend pose
  void ComputePose();

  // update motion model, IMU or Constant model
  void UpdateMotionModel();

  // get current frame[left image] pyramid image
  std::vector<cv::Mat> GetCurrentFramePyramid() const;

  float ComputeParallax(const int kfid, bool do_unrot, bool median, bool is_2donly);

 private:
  MapManagerPtr map_manager_;
  FramePtr current_frame_;
  SettingPtr param_;
  TrackerBasePtr tracker_;
  VisualizationBasePtr viz_;

  ConstantMotionModel motion_model_;

  cv::Mat cur_img_, prev_img_;
  cv::Ptr<cv::CLAHE> clahe_;
  std::vector<cv::Mat> cur_pyr_, prev_pyr_;
  std::vector<cv::Mat> kf_pyr_;

  int klt_win_size_;
  bool use_clahe_ = false;
  bool klt_use_prior_;
  int klt_pyr_level_;
  float klt_err_;
  float klt_max_fb_dist_;
  bool track_keyframetoframe_;
};

typedef std::shared_ptr<VisualFrontEnd> VisualFrontEndPtr;
typedef std::shared_ptr<const VisualFrontEnd> VisualFrontEndConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP
