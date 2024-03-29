#include "vio_hw/internal/map_manager.hpp"

namespace viohw {

MapManager::MapManager(SettingPtr state, FramePtr frame, FeatureBasePtr feat_extract,
                       TrackerBasePtr tracker)
    : param_(state), current_frame_(frame), feature_extractor_(feat_extract), tracker_(tracker) {}

void MapManager::CreateKeyframe(const cv::Mat& im, const cv::Mat& im_raw) {
  PrepareFrame();
  ExtractKeypoints(im, im_raw);
  AddKeyframe();
}

void MapManager::PrepareFrame() {
  // Update new KF id
  current_frame_->kfid_ = kf_id_;
}

void MapManager::AddKeyframe() {  // Create a copy of Cur. Frame shared_ptr for creating an
  // independant KF to add to the map
  std::shared_ptr<Frame> pkf =
      std::allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), *current_frame_);

  std::lock_guard<std::mutex> lock(kf_mutex_);

  // Add KF to the unordered map and update id/nb
  map_kfs_.emplace(kf_id_, pkf);
  lm_id_++;
  kf_id_++;
}

void MapManager::ExtractKeypoints(const cv::Mat& im, const cv::Mat& im_raw) {
  std::vector<Keypoint> kps = current_frame_->getKeypoints();

  std::vector<cv::Point2f> pts;
  std::vector<int> scales;
  std::vector<float> angles;

  for (auto& kp : kps) {
    pts.push_back(kp.px_);
  }

  // TODO brief calculate
  if (param_->feat_tracker_setting_.use_brief_) {
    //.................
  }

  int num_need_detect = param_->feat_tracker_setting_.max_feature_num_ - kps.size();
  if (num_need_detect > 0) {
    // feature_extractor_->detect(im, )
  }

}

}  // namespace viohw
