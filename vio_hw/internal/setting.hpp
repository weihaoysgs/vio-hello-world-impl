
#ifndef VIO_HELLO_WORLD_SETTING_HPP
#define VIO_HELLO_WORLD_SETTING_HPP

#include <glog/logging.h>

#include <Eigen/Core>
#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>

namespace viohw {
class Setting
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  explicit Setting(const std::string& config_file_path);
  Setting() = delete;
  ~Setting() = default;
  void readCameraParams(const cv::FileNode& node);
  void readIMUParams(const cv::FileNode& node);
  void readSLAMParams(const cv::FileNode& node);
  std::string left_image_topic_, right_image_topic_;
  std::vector<std::string> topic_left_right_;
  std::vector<std::string> model_left_right_;
  std::vector<int> left_resolution_, right_resolution_;
  std::vector<float> left_k_, right_k_, left_dist_coeff_, right_dist_coeff_;
  std::string imu_topic_;

  bool stereo_mode_;
  bool use_imu_;
  bool force_realtime_;
};
}  // namespace viohw

#endif  // VIO_HELLO_WORLD_SETTING_HPP
