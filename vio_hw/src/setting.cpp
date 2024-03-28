#include "vio_hw/internal/setting.hpp"

namespace viohw {

Setting::Setting(const std::string& config_file_path) {
  cv::FileStorage fs(config_file_path, cv::FileStorage::READ);
  LOG_IF(FATAL, !fs.isOpened())
      << "Config file open failed, please check your confi file path.";
  cv::FileNode cameras = fs["Camera"];
  cv::FileNode imu = fs["IMU"];
  cv::FileNode slam = fs["SLAM"];
  cv::FileNode feat_tracker = fs["FeatureAndTracker"];
  readCameraParams(cameras);
  readIMUParams(imu);
  readSLAMParams(slam);
  readFeatureTrackerParams(feat_tracker);
  std::cout << cam_setting_ << "\n";
  std::cout << imu_setting_ << "\n";
  std::cout << slam_setting_ << "\n";
  std::cout << feat_tracker_setting_ << "\n";
}

void Setting::readCameraParams(const cv::FileNode& cameras) {
  cameras["topic.left.right"] >> cam_setting_.topic_left_right_;
  cameras["model.left.right"] >> cam_setting_.model_left_right_;
  cameras["left.resolution"] >> cam_setting_.left_resolution_;
  cameras["right.resolution"] >> cam_setting_.right_resolution_;
  cameras["left.K"] >> cam_setting_.left_k_;
  cameras["right.K"] >> cam_setting_.right_k_;
  cameras["left.distortion.coefficient"] >> cam_setting_.left_dist_coeff_;
  cameras["right.distortion.coefficient"] >> cam_setting_.right_dist_coeff_;
}

void Setting::readIMUParams(const cv::FileNode& node) {
  node["topic"] >> imu_setting_.imu_topic_;
  node["acc_n"] >> imu_setting_.acc_n_;
  node["acc_w"] >> imu_setting_.acc_w_;
  node["gyr_n"] >> imu_setting_.gyr_n_;
  node["gyr_w"] >> imu_setting_.gyr_w_;
}

void Setting::readSLAMParams(const cv::FileNode& node) {
  node["stereo.mode"] >> slam_setting_.stereo_mode_;
  node["use.imu"] >> slam_setting_.use_imu_;
  node["force.realtime"] >> slam_setting_.force_realtime_;
  node["use.Pangolin"] >> slam_setting_.use_pangolin_;
  node["use.Rviz"] >> slam_setting_.use_rviz_;
}

void Setting::readFeatureTrackerParams(const cv::FileNode& node) {
  node["max.kps.num"] >> feat_tracker_setting_.max_feature_num_;
  node["max.kps.distance"] >> feat_tracker_setting_.max_feature_dis_;
  node["kp.quality.level"] >> feat_tracker_setting_.feature_quality_level_;
  node["use.clahe"] >> feat_tracker_setting_.use_clahe_;
  node["clahe.val"] >> feat_tracker_setting_.clahe_val_;
}
}  // namespace viohw
