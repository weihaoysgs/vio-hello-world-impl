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

  // TODO
  // for (const auto& kp : current_frame_->getKeypoints()) {
  //   // Get the related MP
  //   auto plmit = map_lms_.find(kp.lmid_);
  //
  //   if (plmit == map_lms_.end()) {
  //     removeObsFromCurFrameById(kp.lmid_);
  //     continue;
  //   }
  //
  //   // Relate new KF id to the MP
  //   plmit->second->addKfObs(nkfid_);
  // }
}

void MapManager::AddKeyframe() {  // Create a copy of Cur. Frame shared_ptr for creating an
  // independant KF to add to the map
  std::shared_ptr<Frame> pkf =
      std::allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), *current_frame_);

  std::lock_guard<std::mutex> lock(kf_mutex_);

  // Add KF to the unordered map and update id/nb
  map_kfs_.emplace(kf_id_, pkf);
  num_kfs_++;
  kf_id_++;
}

void MapManager::ExtractKeypoints(const cv::Mat& im, const cv::Mat& im_raw) {
  std::vector<Keypoint> kps = current_frame_->getKeypoints();

  // TODO brief calculate
  if (param_->feat_tracker_setting_.use_brief_) {
    //.................
  }

  int num_need_detect = param_->feat_tracker_setting_.max_feature_num_ - kps.size();
  if (num_need_detect > 0) {
    std::vector<cv::KeyPoint> new_kps;
    feature_extractor_->setMaxKpsNumber(num_need_detect);
    feature_extractor_->detect(im, new_kps);
    if (!new_kps.empty()) {
      std::vector<cv::Point2f> desc_pts;
      cv::KeyPoint::convert(new_kps, desc_pts);
      if (param_->feat_tracker_setting_.use_brief_) {
        std::vector<cv::Mat> vdescs = feature_extractor_->DescribeBRIEF(im_raw, desc_pts);
        AddKeypointsToFrame(desc_pts, vdescs, *current_frame_);
      } else {
        AddKeypointsToFrame(desc_pts, *current_frame_);
      }
    }
  }
}

void MapManager::AddKeypointsToFrame(const std::vector<cv::Point2f>& vpts, Frame& frame) {
  std::lock_guard<std::mutex> lock(lm_mutex_);

  // Add keypoints + create MPs
  for (const auto& vpt : vpts) {
    // Add keypoint to current frame
    frame.AddKeypoint(vpt, lm_id_);
    // Create landmark with same id
    AddMapPoint();
  }
}

void MapManager::AddKeypointsToFrame(const std::vector<cv::Point2f>& vpts,
                                     const std::vector<cv::Mat>& vdescs, Frame& frame) {
  std::lock_guard<std::mutex> lock(lm_mutex_);

  // Add keypoints + create landmarks
  for (size_t i = 0; i < vpts.size(); i++) {
    if (!vdescs.at(i).empty()) {
      // Add keypoint to current frame
      frame.AddKeypoint(vpts.at(i), lm_id_, vdescs.at(i));
      // Create landmark with same id
      AddMapPoint(vdescs.at(i));
    } else {
      // Add keypoint to current frame
      frame.AddKeypoint(vpts.at(i), lm_id_);
      // Create landmark with same id
      AddMapPoint();
    }
  }
}

void MapManager::AddMapPoint() {
  // Create a new MP with a unique lmid and a KF id obs
  std::shared_ptr<MapPoint> plm =
      std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), lm_id_, kf_id_);

  // Add new MP to the map and update id/nb
  map_lms_.emplace(lm_id_, plm);
  lm_id_++;
  num_lms_++;

  // Visualization related part for pointcloud obs
  // TODO
}

void MapManager::AddMapPoint(const cv::Mat& desc) {
  // Create a new MP with a unique lmid and a KF id obs
  std::shared_ptr<MapPoint> plm =
      std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), lm_id_, kf_id_, desc);

  // Add new MP to the map and update id/nb
  map_lms_.emplace(lm_id_, plm);
  lm_id_++;
  num_lms_++;

  // Visualization related part for pointcloud obs
  // TODO
}

// Remove a MP obs from cur Frame
void MapManager::RemoveObsFromCurFrameById(const int lmid)
{
  // Remove cur obs
  current_frame_->RemoveKeypointById(lmid);

  // Set MP as not obs
  // TODO
}

}  // namespace viohw
