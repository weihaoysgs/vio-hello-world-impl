#include "vio_hw/internal/world_manager.hpp"

#include "common/draw_utils.h"
#include "vio_hw/internal/feat/good_feature_impl.hpp"

namespace viohw {
WorldManager::WorldManager(std::shared_ptr<Setting>& setting)
    : params_(setting) {
  feature_extractor_ =
      FeatureBase::Create(GoodFeature2Tracker::getDefaultOptions());
}

void WorldManager::run() {
  cv::Mat img_left, img_right;
  double cur_time;
  while (true) {
    if (getNewImage(img_left, img_right, cur_time)) {
      std::vector<cv::KeyPoint> kps;
      feature_extractor_->detect(img_left, kps);

      cv::imshow("image0", com::DrawKeyPoints(img_left, kps));
      cv::waitKey(1);
    }
    std::chrono::milliseconds dura(1);
    std::this_thread::sleep_for(dura);
  }
}

void WorldManager::addNewStereoImages(const double time, cv::Mat& im0,
                                      cv::Mat& im1) {
  std::lock_guard<std::mutex> lock(img_mutex_);
  img_left_queen_.push(im0);
  img_right_queen_.push(im1);
  img_time_queen_.push(time);

  is_new_img_available_ = true;
}

bool WorldManager::getNewImage(cv::Mat& iml, cv::Mat& imr, double& time) {
  std::lock_guard<std::mutex> lock(img_mutex_);

  if (!is_new_img_available_) {
    return false;
  }
  int k = 0;

  do {
    k++;

    iml = img_left_queen_.front();
    img_left_queen_.pop();

    time = img_time_queen_.front();
    img_time_queen_.pop();

    if (params_->stereo_mode_) {
      imr = img_right_queen_.front();
      img_right_queen_.pop();
    }

    // if not force realtime, will process every frame
    if (!params_->force_realtime_) {
      break;
    }

  } while (!img_left_queen_.empty());

  if (k > 1) {
    LOG(WARNING) << " SLAM is late! Skipped " << k - 1 << " frames...\n";
  }

  if (img_left_queen_.empty()) {
    is_new_img_available_ = false;
  }

  return true;
}
}  // namespace viohw