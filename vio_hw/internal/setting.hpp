
#ifndef VIO_HELLO_WORLD_SETTING_HPP
#define VIO_HELLO_WORLD_SETTING_HPP

#include <glog/logging.h>

#include <Eigen/Core>
#include <atomic>
#include <iomanip>
#include <memory>
#include <opencv2/opencv.hpp>

#include "common/print_tools.hpp"

namespace viohw {
class Setting
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  struct CameraSetting
  {
    std::string name_ = "{CameraSetting}";
    std::string left_image_topic_, right_image_topic_;
    std::vector<std::string> topic_left_right_;
    std::vector<std::string> model_left_right_;
    std::vector<int> left_resolution_, right_resolution_;
    std::vector<float> left_k_, right_k_, left_dist_coeff_, right_dist_coeff_;
  };

  struct FeatureAndTrackerSetting
  {
    std::string name_ = "{FeatureTrackerSetting}";
    int max_feature_num_;
    int max_feature_dis_;
    double feature_quality_level_;
    bool use_clahe_;
    double clahe_val_;
    bool klt_use_prior_;
    bool track_keyframetoframe_;
    int klt_win_size_;
    int klt_pyr_level_;
    bool use_brief_;
  };

  struct IMUSetting
  {
    std::string name_ = "{IMUSetting}";
    std::string imu_topic_;
    double acc_n_, acc_w_;
    double gyr_n_, gyr_w_;
  };

  struct SLAMSetting
  {
    std::string name_ = "{SLAMSetting}";
    bool stereo_mode_;
    bool use_imu_;
    bool force_realtime_;
    bool use_rviz_;
    bool use_pangolin_;
  };

  explicit Setting(const std::string& config_file_path);
  Setting() = delete;
  ~Setting() = default;
  void readCameraParams(const cv::FileNode& node);
  void readIMUParams(const cv::FileNode& node);
  void readSLAMParams(const cv::FileNode& node);
  void readFeatureTrackerParams(const cv::FileNode& node);

  CameraSetting cam_setting_;
  FeatureAndTrackerSetting feat_tracker_setting_;
  IMUSetting imu_setting_;
  SLAMSetting slam_setting_;
};

typedef std::shared_ptr<Setting> SettingPtr;
typedef std::shared_ptr<const Setting> SettingConstPtr;

// clang-format off
inline std::ostream& operator<<(std::ostream& os,
                                const Setting::IMUSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::right << std::setw(20) << "IMU Topic: "       << std::left << std::setw(10) << setting.imu_topic_ << std::endl
     << std::right << std::setw(20) << "Acc Noise: "       << std::left << std::setw(10) << setting.acc_n_ << std::endl
     << std::right << std::setw(20) << "Acc Random Walk: " << std::left << std::setw(10) << setting.acc_w_ << std::endl
     << std::right << std::setw(20) << "Gyr Noise: "       << std::left << std::setw(10) << setting.gyr_n_ << std::endl
     << std::right << std::setw(20) << "Gyr Random Walk: " << std::left << std::setw(10) << setting.gyr_w_;
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os, const Setting::FeatureAndTrackerSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::boolalpha;
  os << std::right << std::setw(20) << "Max.Feature.Num: " << std::left << std::setw(10) <<  setting.max_feature_num_ << std::endl
     << std::right << std::setw(20) << "Max.Feature.Dis: " << std::left << std::setw(10) <<  setting.max_feature_dis_ << std::endl
     << std::right << std::setw(20) << "Quality.Level: "   << std::left << std::setw(10) <<  setting.feature_quality_level_ << std::endl
     << std::right << std::setw(20) << "Use.CLAHE: "       << std::left << std::setw(10) <<  (setting.use_clahe_ ? "true" : "false") << std::endl
     << std::right << std::setw(20) << "CLAHE.Val: "       << std::left << std::setw(10) <<  setting.clahe_val_ << std::endl
     << std::right << std::setw(20) << "KLT.Pyra.Level: "  << std::left << std::setw(10) <<  setting.klt_pyr_level_ << std::endl
     << std::right << std::setw(20) << "KLT.Win.Size: "    << std::left << std::setw(10) <<  setting.klt_win_size_ << std::endl
     << std::right << std::setw(20) << "KLT.Use.Prior: "   << std::left << std::setw(10) <<  setting.klt_use_prior_ << std::endl
     << std::right << std::setw(20) << "Use.Brief: "       << std::left << std::setw(10) <<  setting.use_brief_ << std::endl
     << std::right << std::setw(20) << "Track.KF2Frame: "  << std::left << std::setw(10) <<  setting.track_keyframetoframe_ ;
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const Setting::SLAMSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::boolalpha
     << std::right << std::setw(20)  << "Stereo Mode: "    << std::left << std::setw(10) << setting.stereo_mode_    << std::endl
     << std::right << std::setw(20)  << "Use IMU: "        << std::left << std::setw(10) << setting.use_imu_        << std::endl
     << std::right << std::setw(20)  << "Force Realtime: " << std::left << std::setw(10) << setting.force_realtime_ << std::endl
     << std::right << std::setw(20)  << "Use RViz: "       << std::left << std::setw(10) << setting.use_rviz_       << std::endl
     << std::right << std::setw(20)  << "Use Pangolin: "   << std::left << std::setw(10) << setting.use_pangolin_;
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const Setting::CameraSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::right << std::setw(20) << "Topic Left-Right: ";
  for (const auto& topic : setting.topic_left_right_) {
    os << topic << " ";
  }
  os << std::endl;
  os << std::right << std::setw(20) << "Model Left-Right: ";
  for (const auto& model : setting.model_left_right_) {
    os << model << " ";
  }
  os << std::endl;
  os << std::right << std::setw(20) 
     << "Left Resolution: " << setting.left_resolution_[0] << "x"
     << setting.left_resolution_[1] << std::endl
     << std::right << std::setw(20)
     << "Right Resolution: " << setting.right_resolution_[0] << "x"
     << setting.right_resolution_[1] << std::endl
     << std::right << std::setw(20)
     << "Left K: ";
  for (const auto& k : setting.left_k_) {
    os << k << " ";
  }
  os << std::endl;
  os << std::right << std::setw(20) << "Right K: ";
  for (const auto& k : setting.right_k_) {
    os << k << " ";
  }
  os << std::endl;
  os << std::right << std::setw(20) << "Left Dis Coeff: ";
  for (const auto& coeff : setting.left_dist_coeff_) {
    os << coeff << " ";
  }
  os << std::endl;
  os << std::right << std::setw(20) << "Right Dis Coeff: ";
  for (const auto& coeff : setting.right_dist_coeff_) {
    os << coeff << " ";
  }
  return os;
}
// clang-format on
}  // namespace viohw

#endif  // VIO_HELLO_WORLD_SETTING_HPP
