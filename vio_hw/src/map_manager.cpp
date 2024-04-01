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
  std::vector<viohw::Keypoint> kps = current_frame_->getKeypoints();

  // TODO brief calculate
  if (param_->feat_tracker_setting_.use_brief_) {
    //.................
  }

  cv::Mat mask = cv::Mat(im.rows, im.cols, CV_8UC1, cv::Scalar(255));
  for (auto& pt : kps) {
    cv::circle(mask, pt.px_, param_->feat_tracker_setting_.max_feature_dis_, 0, -1);
  }

  int num_need_detect = param_->feat_tracker_setting_.max_feature_num_ - kps.size();
  if (num_need_detect > 0) {
    std::vector<cv::KeyPoint> new_kps;
    feature_extractor_->setMaxKpsNumber(num_need_detect);
    feature_extractor_->detect(im, new_kps, mask);
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
void MapManager::RemoveObsFromCurFrameById(const int lmid) {
  // Remove cur obs
  current_frame_->RemoveKeypointById(lmid);

  // Set MP as not obs
  // TODO
}

std::shared_ptr<Frame> MapManager::GetKeyframe(const int kfid) const {
  std::lock_guard<std::mutex> lock(kf_mutex_);

  auto it = map_kfs_.find(kfid);
  if (it == map_kfs_.end()) {
    return nullptr;
  }
  return it->second;
}

std::shared_ptr<MapPoint> MapManager::GetMapPoint(const int lmid) const {
  std::lock_guard<std::mutex> lock(lm_mutex_);

  auto it = map_lms_.find(lmid);
  if (it == map_lms_.end()) {
    return nullptr;
  }
  return it->second;
}

void MapManager::StereoMatching(Frame& frame, const std::vector<cv::Mat>& vleftpyr,
                                const std::vector<cv::Mat>& vrightpyr) {
  // Find stereo correspondances with left kps
  auto vleftkps = frame.getKeypoints();
  size_t nbkps = vleftkps.size();

  std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
  std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
  for (size_t i = 0; i < nbkps; i++) {
    // Set left kp
    auto& kp = vleftkps.at(i);

    // Set prior right kp
    cv::Point2f priorpt = kp.px_;

    vkpids.push_back(kp.lmid_);
    vkps.push_back(kp.px_);
    vpriors.push_back(priorpt);
  }

  std::vector<cv::Point2f> good_right_kps;
  std::vector<int> good_ids;
  size_t num_good = 0, tracker_good = 0, inliner_good = 0;
  if (!vkps.empty()) {
    // Good / bad kps vector
    std::vector<bool> vkpstatus;
    std::vector<uchar> inliers;
    tracker_->trackerAndMatcher(
        vleftpyr, vrightpyr, param_->feat_tracker_setting_.klt_win_size_,
        param_->feat_tracker_setting_.klt_pyr_level_, param_->feat_tracker_setting_.klt_err_,
        param_->feat_tracker_setting_.klt_max_fb_dist_, vkps, vpriors, vkpstatus);


    for (size_t i = 0; i < vkpstatus.size(); i++) {
      if (vkpstatus.at(i)) {
        frame.UpdateKeypointStereo(vkpids.at(i), vpriors.at(i));
        num_good++;
      }
    }
    // TODO
  }
  LOG(INFO) << "kp num: " << vkps.size() << ",Good Stereo Matching Num: " << num_good
            << ",tracker good:" << tracker_good << ", inliner good:" << inliner_good;
}

void MapManager::UpdateMapPoint(const int lmid, const Eigen::Vector3d& wpt,
                                const double inv_depth) {
  std::lock_guard<std::mutex> lock(lm_mutex_);
  std::lock_guard<std::mutex> lock_kf(kf_mutex_);
  auto plmit = map_lms_.find(lmid);

  if (plmit == map_lms_.end()) {
    return;
  }
  if (plmit->second == nullptr) {
    return;
  }
  // If MP 2D -> 3D => Notif. KFs
  if (!plmit->second->is3d_) {
    for (const auto& kfid : plmit->second->GetKfObsSet()) {
      auto pkfit = map_kfs_.find(kfid);
      if (pkfit != map_kfs_.end()) {
        pkfit->second->TurnKeypoint3d(lmid);
      } else {
        plmit->second->RemoveKfObs(kfid);
      }
    }
    if (plmit->second->isobs_) {
      current_frame_->TurnKeypoint3d(lmid);
    }
  }

  // Update MP world pos.
  if( inv_depth >= 0. ) {
    plmit->second->SetPoint(wpt, inv_depth);
  } else {
    plmit->second->SetPoint(wpt);
  }
}
// Remove a KF obs from a MP
void MapManager::RemoveMapPointObs(const int lmid, const int kfid) {
  std::lock_guard<std::mutex> lock(lm_mutex_);
  std::lock_guard<std::mutex> lockkf(kf_mutex_);

  // Remove MP obs from KF
  auto pkfit = map_kfs_.find(kfid);
  if (pkfit != map_kfs_.end()) {
    pkfit->second->RemoveKeypointById(lmid);
  }

  // Remove KF obs from MP
  auto plmit = map_lms_.find(lmid);

  // Skip if MP does not exist
  if (plmit == map_lms_.end()) {
    return;
  }
  plmit->second->RemoveKfObs(kfid);

  // TODO
  // if( pkfit != map_pkfs_.end() ) {
  //   for( const auto &cokfid : plmit->second->getKfObsSet() ) {
  //     auto pcokfit = map_pkfs_.find(cokfid);
  //     if( pcokfit != map_pkfs_.end() ) {
  //       pkfit->second->decreaseCovisibleKf(cokfid);
  //       pcokfit->second->decreaseCovisibleKf(kfid);
  //     }
  //   }
  // }
}
}  // namespace viohw
