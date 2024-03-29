#include "vio_hw/internal/world_manager.hpp"

namespace viohw {

WorldManager::WorldManager(std::shared_ptr<Setting>& setting)
    : params_(setting) {
  setupCalibration();

  // create feature extractor
  auto feature_options = FeatureBase::getDefaultOptions();
  feature_extractor_ = FeatureBase::Create(feature_options);

  // create visualization
  VisualizationBase::VisualizationOption viz_option{
      VisualizationBase::PANGOLIN};
  viz_ = VisualizationBase::Create(viz_option);

  // create feature tracker
  TrackerBase::TrackerOption tracker_option{TrackerBase::LIGHT_GLUE};
  tracker_ = TrackerBase::Create(tracker_option);

  // TODO: for [ncellsize] param
  // create current frame
  if (!params_->slam_setting_.stereo_mode_) {
    current_frame_.reset(new Frame(calib_model_left_, 35));
  } else {
    current_frame_.reset(new Frame(calib_model_left_, calib_model_right_, 35));
  }

  // create map manager
  map_manager_.reset(
      new MapManager(params_, current_frame_, feature_extractor_, tracker_));

  // create visual frontend
  visual_frontend_.reset(
      new VisualFrontEnd(params_, current_frame_, map_manager_, tracker_));
}

void WorldManager::run() {
  cv::Mat img_left, img_right;
  double cur_time;
  while (true) {
    if (getNewImage(img_left, img_right, cur_time)) {
      frame_id_++;
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

    if (params_->slam_setting_.stereo_mode_) {
      imr = img_right_queen_.front();
      img_right_queen_.pop();
    }

    // if not force realtime, will process every frame
    if (!params_->slam_setting_.force_realtime_) {
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

void WorldManager::setupCalibration() {
  // clang-format off
  calib_model_left_.reset(new CameraCalibration(
      params_->cam_setting_.model_left_right_[0],
      params_->cam_setting_.left_k_[0], 
      params_->cam_setting_.left_k_[1],
      params_->cam_setting_.left_k_[2], 
      params_->cam_setting_.left_k_[3], 
      params_->cam_setting_.left_dist_coeff_[0],
      params_->cam_setting_.left_dist_coeff_[1], 
      params_->cam_setting_.left_dist_coeff_[2],
      params_->cam_setting_.left_dist_coeff_[3], 
      params_->cam_setting_.left_resolution_[0],
      params_->cam_setting_.left_resolution_[1]));

  if (params_->slam_setting_.stereo_mode_) {
    calib_model_right_.reset(new CameraCalibration(
        params_->cam_setting_.model_left_right_[1], 
        params_->cam_setting_.right_k_[0],
        params_->cam_setting_.right_k_[1], 
        params_->cam_setting_.right_k_[2], 
        params_->cam_setting_.right_k_[3],
        params_->cam_setting_.right_dist_coeff_[0], 
        params_->cam_setting_.right_dist_coeff_[1],
        params_->cam_setting_.right_dist_coeff_[2], 
        params_->cam_setting_.right_dist_coeff_[3],
        params_->cam_setting_.right_resolution_[0], 
        params_->cam_setting_.right_resolution_[1]));
    // clang-format on
    // TODO: Change this and directly add the extrinsic parameters within the
    // constructor (maybe set default parameters on extrinsic with identity /
    // zero)
    // calib_model_right_->setupExtrinsic(params_->T_left_right_);
  }
}
}  // namespace viohw