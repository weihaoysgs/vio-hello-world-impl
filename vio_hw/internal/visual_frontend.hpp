#ifndef VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP
#define VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP

#include "vio_hw/internal/constant_motion_model.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"

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
                 TrackerBasePtr tracker);

  // tracking left image
  bool TrackerMono(cv::Mat &image, double time);

  // tracking image and build keyframe
  bool VisualTracking(cv::Mat &image, double time);

  // preprocess image, build pyramid for tracking and apply clahe
  void PreProcessImage(cv::Mat &image);

 private:
  MapManagerPtr map_manager_;
  FramePtr current_frame_;
  SettingPtr param_;
  TrackerBasePtr tracker_;

  ConstantMotionModel motion_model_;

  cv::Mat cur_img_, prev_img_;
  cv::Ptr<cv::CLAHE> clahe_;
  std::vector<cv::Mat> cur_pyr_, prev_pyr_;
  std::vector<cv::Mat> kf_pyr_;

  cv::Size klt_win_size_;
  bool use_clahe_ = false;
};

typedef std::shared_ptr<VisualFrontEnd> VisualFrontEndPtr;
typedef std::shared_ptr<const VisualFrontEnd> VisualFrontEndConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP
