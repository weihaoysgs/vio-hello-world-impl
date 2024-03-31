#include "vio_hw/internal/visual_frontend.hpp"

namespace viohw {

VisualFrontEnd::VisualFrontEnd(viohw::SettingPtr state, viohw::FramePtr frame,
                               viohw::MapManagerPtr map, viohw::TrackerBasePtr tracker,
                               VisualizationBasePtr viz)
    : param_(state), current_frame_(frame), map_manager_(map), tracker_(tracker), viz_(viz) {
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
  // preprocess
  PreProcessImage(image);

  // first frame is keyframe
  if (current_frame_->id_ == 0) {
    return true;
  }

  // tracking from frame to frame
  KLTTracking();

  // outlier filter
  Epipolar2d2dFiltering();

  // show tracking result to ui
  ShowTrackingResult();

  // compute current visual frontend pose
  ComputePose();

  // update motion model
  UpdateMotionModel();

  // check is new keyframe
  return CheckIsNewKeyframe();
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

void VisualFrontEnd::Epipolar2d2dFiltering() {
  // Get prev KF
  auto pkf = map_manager_->GetKeyframe(current_frame_->kfid_);

  if (pkf == nullptr) {
    LOG(FATAL) << "ERROR! Previous Kf does not exist yet (epipolar2d2d()).";
  }

  // Get cur. Frame nb kps
  size_t nbkps = current_frame_->nbkps_;

  if (nbkps < 8) {
    LOG(WARNING) << "Not enough kps to compute Essential Matrix";
    return;
  }
  std::vector<int> vkpsids;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vkfbvs, vcurbvs;
  std::vector<cv::Point2f> vkf_px, vcur_px;

  for (const auto& it : current_frame_->mapkps_) {
    auto& kp = it.second;

    // Get the prev. KF related kp if it exists
    auto kf_kp = pkf->GetKeypointById(kp.lmid_);

    if (kf_kp.lmid_ != kp.lmid_) {
      continue;
    }
    vkfbvs.push_back(kf_kp.bv_);
    vcurbvs.push_back(kp.bv_);
    vkf_px.push_back(kf_kp.px_);
    vcur_px.push_back(kp.px_);
    vkpsids.push_back(kp.lmid_);
    // TODO
  }
  std::vector<uchar> inliers;
  cv::findFundamentalMat(vkf_px, vcur_px, cv::FM_RANSAC, 3, 0.99, inliers);
  assert(vkf_px.size() == vcur_px.size() && vcur_px.size() == inliers.size());
  for (size_t i = 0; i < inliers.size(); i++) {
    if (!inliers[i]) {
      map_manager_->RemoveObsFromCurFrameById(vkpsids[i]);
    }
  }
}

void VisualFrontEnd::ShowTrackingResult() {
  cv::Mat draw_tracker_image;
  cv::cvtColor(cur_img_, draw_tracker_image, cv::COLOR_GRAY2BGR);
  // Get prev KF
  auto pkf = map_manager_->GetKeyframe(current_frame_->kfid_);
  std::vector<cv::Point2f> vkf_px, vcur_px;
  for (const auto& it : current_frame_->mapkps_) {
    auto& kp = it.second;
    // Get the prev. KF related kp if it exists
    auto kf_kp = pkf->GetKeypointById(kp.lmid_);
    if (kf_kp.lmid_ != kp.lmid_) {
      continue;
    }
    vkf_px.push_back(kf_kp.px_);
    vcur_px.push_back(kp.px_);
    cv::arrowedLine(draw_tracker_image, kp.px_, kf_kp.px_, cv::Scalar(0, 255, 0), 2, 8, 0, 0.3);
    cv::circle(draw_tracker_image, kf_kp.px_, 2, cv::Scalar(0, 255, 0), -1);
  }
  // cv::imshow("tracker", draw_tracker_image);
  // cv::waitKey(1);
  viz_->showTrackerResultImage(draw_tracker_image);
}

bool VisualFrontEnd::CheckIsNewKeyframe() { return true; }

void VisualFrontEnd::ComputePose() {}

void VisualFrontEnd::UpdateMotionModel() {}

}  // namespace viohw
