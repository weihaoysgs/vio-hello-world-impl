#include "vio_hw/internal/visual_frontend.hpp"

namespace viohw {

VisualFrontEnd::VisualFrontEnd(viohw::SettingPtr state, viohw::FramePtr frame,
                               viohw::MapManagerPtr map, viohw::TrackerBasePtr tracker)
    : param_(state), current_frame_(frame), map_manager_(map), tracker_(tracker) {
  use_clahe_ = param_->feat_tracker_setting_.use_clahe_;

  if (use_clahe_) {
    int tilesize = 50;
    cv::Size clahe_tiles(param_->cam_setting_.left_resolution_[0] / tilesize,
                         param_->cam_setting_.left_resolution_[1] / tilesize);
    clahe_ = cv::createCLAHE(param_->feat_tracker_setting_.clahe_val_, clahe_tiles);
  }
  klt_use_prior_ = param_->feat_tracker_setting_.klt_use_prior_;
  klt_win_size_ = param_->feat_tracker_setting_.klt_win_size_;
  klt_pyr_level_ = param_->feat_tracker_setting_.klt_pyr_level_;
  klt_err_ = param_->feat_tracker_setting_.klt_err_;
  klt_max_fb_dist_ = param_->feat_tracker_setting_.klt_max_fb_dist_;
  track_keyframetoframe_ = param_->feat_tracker_setting_.track_keyframetoframe_;
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

  KLTTracking();
  // Only For Test.
  return true;

  return false;
}

void VisualFrontEnd::PreProcessImage(cv::Mat& img_raw) {
  if (use_clahe_) {
    clahe_->apply(img_raw, cur_img_);
  } else {
    cur_img_ = img_raw;
  }

  if (!cur_pyr_.empty() && !track_keyframetoframe_) {
    prev_pyr_.swap(cur_pyr_);
  }

  cv::buildOpticalFlowPyramid(cur_img_, cur_pyr_, cv::Size(klt_win_size_, klt_win_size_),
                              param_->feat_tracker_setting_.klt_pyr_level_);
}

void VisualFrontEnd::KLTTracking() {
  // Get current kps and init priors for tracking
  std::vector<int> v3d_kp_ids, vkp_ids;
  std::vector<cv::Point2f> v3d_kps, v3d_priors, vkps, vpriors;
  std::vector<bool> vkp_is3d;

  // First we're gonna track 3d kps on only 2 levels
  v3d_kp_ids.reserve(current_frame_->nb3dkps_);
  v3d_kps.reserve(current_frame_->nb3dkps_);
  v3d_priors.reserve(current_frame_->nb3dkps_);

  // Then we'll track 2d kps on full pyramid levels
  vkp_ids.reserve(current_frame_->nbkps_);
  vkps.reserve(current_frame_->nbkps_);
  vpriors.reserve(current_frame_->nbkps_);

  vkp_is3d.reserve(current_frame_->nbkps_);

  // Front-End is thread-safe so we can direclty access curframe's kps
  for (const auto& it : current_frame_->mapkps_) {
    auto& kp = it.second;

    // Init prior px pos. from motion model
    if (klt_use_prior_) {
      if (kp.is3d_) {
        // TODO
      }
    }

    // For other kps init prior with prev px pos.
    vkp_ids.push_back(kp.lmid_);
    vkps.push_back(kp.px_);
    vpriors.push_back(kp.px_);
  }

  // 1st track 3d kps if using prior
  if (klt_use_prior_ && !v3d_priors.empty()) {
    // TODO
  }
  // 2st tracker 2d kps
  if (!vkps.empty()) {
    // Good / bad kps vector
    std::vector<bool> kpstatus;

    tracker_->trackerAndMatcher(prev_pyr_, cur_pyr_, klt_win_size_, klt_pyr_level_, klt_err_,
                                klt_max_fb_dist_, vkps, vpriors, kpstatus);

    size_t good_num = 0;

    for (size_t i = 0; i < vkps.size(); i++) {
      if (kpstatus.at(i)) {
        current_frame_->UpdateKeypoint(vkp_ids.at(i), vpriors.at(i));
        good_num++;
      } else {
        // MapManager is responsible for all the removing operations
        map_manager_->RemoveObsFromCurFrameById(vkp_ids.at(i));
      }
    }
  }
}

}  // namespace viohw
