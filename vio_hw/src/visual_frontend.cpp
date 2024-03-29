#include "vio_hw/internal/visual_frontend.hpp"

namespace viohw {

VisualFrontEnd::VisualFrontEnd(viohw::SettingPtr state, viohw::FramePtr frame,
                               viohw::MapManagerPtr map, viohw::TrackerBasePtr tracker)
    : param_(state), current_frame_(frame), map_manager_(map), tracker_(tracker) {
  use_clahe_ = param_->feat_tracker_setting_.use_clahe_;
  int win_size = param_->feat_tracker_setting_.klt_win_size_;
  klt_win_size_ = cv::Size(win_size, win_size);

  std::cout << std::endl
            << std::right << std::setw(40) << "VisualFrontEnd::klt_win_size_: " << std::left
            << klt_win_size_ << std::endl;

  if (use_clahe_) {
    int tilesize = 50;
    cv::Size clahe_tiles(param_->cam_setting_.left_resolution_[0] / tilesize,
                         param_->cam_setting_.left_resolution_[1] / tilesize);
    clahe_ = cv::createCLAHE(param_->feat_tracker_setting_.clahe_val_, clahe_tiles);
  }
}

bool VisualFrontEnd::VisualTracking(cv::Mat& image, double time) {
  bool is_kf = TrackerMono(image, time);
  if (is_kf) {
    map_manager_->CreateKeyframe(cur_img_, image);
  }

  return is_kf;
}

bool VisualFrontEnd::TrackerMono(cv::Mat& image, double time) {
  PreProcessImage(image);

  if (current_frame_->id_ == 0) {
    return true;
  }

  return false;
}

void VisualFrontEnd::PreProcessImage(cv::Mat& img_raw) {
  if (param_->feat_tracker_setting_.use_clahe_) {
    clahe_->apply(img_raw, cur_img_);
  } else {
    cur_img_ = img_raw;
  }

  cv::buildOpticalFlowPyramid(cur_img_, cur_pyr_, klt_win_size_,
                              param_->feat_tracker_setting_.klt_pyr_level_);
}

}  // namespace viohw
