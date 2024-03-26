#include "vio_hw/internal/setting.hpp"

namespace viohw {

Setting::Setting(const std::string& config_file_path) {
  cv::FileStorage fs(config_file_path, cv::FileStorage::READ);
  LOG_IF(FATAL, !fs.isOpened())
      << "Config file open failed, please check your confi file path.";
  cv::FileNode cameras = fs["Camera"];
  cv::FileNode imu = fs["IMU"];
  cv::FileNode slam = fs["SLAM"];
  readCameraParams(cameras);
  readIMUParams(imu);
  readSLAMParams(slam);
}

void Setting::readCameraParams(const cv::FileNode& cameras) {
  cameras["topic.left.right"] >> topic_left_right_;
  cameras["model.left.right"] >> model_left_right_;
  cameras["left.resolution"] >> left_resolution_;
  cameras["right.resolution"] >> right_resolution_;
  cameras["left.K"] >> left_k_;
  cameras["right.K"] >> right_k_;
  cameras["left.distortion.coefficient"] >> left_dist_coeff_;
  cameras["right.distortion.coefficient"] >> right_dist_coeff_;

  std::cout << "Left Camera Topic: " << topic_left_right_[0] << std::endl;
  std::cout << "Right Camera Topic: " << topic_left_right_[1] << std::endl;

  std::cout << "Left Camera Model: " << model_left_right_[0] << std::endl;
  std::cout << "Right Camera Model: " << model_left_right_[1] << std::endl;

  std::cout << "Left Camera Resolution: " << left_resolution_[0] << "x"
            << left_resolution_[1] << std::endl;
  std::cout << "Right Camera Resolution: " << right_resolution_[0] << "x"
            << right_resolution_[1] << std::endl;

  std::cout << "Left Camera Intrinsic Matrix K: ";
  for (auto val : left_k_) std::cout << val << " ";
  std::cout << std::endl;

  std::cout << "Right Camera Intrinsic Matrix K: ";
  for (auto val : right_k_) std::cout << val << " ";
  std::cout << std::endl;

  std::cout << "Left Camera Distortion Coefficients: ";
  for (auto val : left_dist_coeff_) std::cout << val << " ";
  std::cout << std::endl;

  std::cout << "Right Camera Distortion Coefficients: ";
  for (auto val : right_dist_coeff_) std::cout << val << " ";
  std::cout << std::endl;
}

void Setting::readIMUParams(const cv::FileNode& node) {
  node["topic"] >> imu_topic_;
  std::cout << "IMU topic: " << imu_topic_ << std::endl;
}
void Setting::readSLAMParams(const cv::FileNode& node) {
  node["stereo.mode"] >> stereo_mode_;
  node["use.imu"] >> use_imu_;
  node["force.realtime"] >> force_realtime_;
  std::cout << "Stereo Mode: " << stereo_mode_ << std::endl;
  std::cout << "Use IMU: " << use_imu_ << std::endl;
  std::cout << "Force Realtime: " << force_realtime_ << std::endl;
}
}  // namespace viohw
