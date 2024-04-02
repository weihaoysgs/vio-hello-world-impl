
#ifndef VIO_HELLO_WORLD_SETTING_HPP
#define VIO_HELLO_WORLD_SETTING_HPP

#include <glog/logging.h>

#include <Eigen/Core>
#include <atomic>
#include <iomanip>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "common/print_tools.hpp"
#include "sophus/se3.hpp"

namespace viohw {
class Setting
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // camera related parameters
  struct CameraSetting
  {
    std::string name_ = "{CameraSetting}";
    std::string left_image_topic_, right_image_topic_;
    std::vector<std::string> topic_left_right_;
    std::vector<std::string> model_left_right_;
    std::vector<int> left_resolution_, right_resolution_;
    std::vector<float> left_k_, right_k_, left_dist_coeff_, right_dist_coeff_;
  };

  // feature extraction and tracking related parameters
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
    int klt_max_iter_;
    float klt_max_px_precision_;
    float klt_max_fb_dist_;
    float klt_err_;
    bool use_brief_;
  };

  // extrinsic parameters
  struct ExtrinsicSetting
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    std::string name_ = "{ExtrinsicSetting}";
    int camera_num_ = 2;
    Sophus::SE3d Tbc0_, Tbc1_;
    Sophus::SE3d Tbc2_, Tbc3_;
    Sophus::SE3d Tc0c1_;
    Sophus::SE3d T_left_right_;
  };

  // IMU parameters
  struct IMUSetting
  {
    std::string name_ = "{IMUSetting}";
    std::string imu_topic_;
    double acc_n_, acc_w_;
    double gyr_n_, gyr_w_;
  };

  // slam parameters
  struct SLAMSetting
  {
    std::string name_ = "{SLAMSetting}";
    bool stereo_mode_;
    bool use_imu_;
    bool force_realtime_;
    bool use_rviz_;
    bool use_pangolin_;
  };

  struct LoopCloserSetting
  {
    std::string name_ = "LoopSetting";
    bool use_loop_closer_;
    double loop_threshold_;
  };

  // explicit construction function
  explicit Setting( const std::string& config_file_path );

  // delete default construction
  Setting() = delete;

  // default de-construction function
  ~Setting() = default;

  // read all parameters
  void Init( const cv::FileStorage& fs );

  // read camera related params
  void readCameraParams( const cv::FileNode& node );

  // read imu related params
  void readIMUParams( const cv::FileNode& node );

  // read slam related params
  void readSLAMParams( const cv::FileNode& node );

  // read feature & tracker related params
  void readFeatureTrackerParams( const cv::FileNode& node );

  // read extrinsic params
  void readExtrinsicParams( const cv::FileNode& node );

  // read extrinsic params
  void readLoopCloserParams( const cv::FileNode& node );

  CameraSetting cam_setting_;
  FeatureAndTrackerSetting feat_tracker_setting_;
  IMUSetting imu_setting_;
  SLAMSetting slam_setting_;
  ExtrinsicSetting extrinsic_setting_;
  LoopCloserSetting loop_setting_;
};

typedef std::shared_ptr<Setting> SettingPtr;
typedef std::shared_ptr<const Setting> SettingConstPtr;

// Overloaded operators <<
// clang-format off
inline std::ostream& operator<<(std::ostream& os,
                                const Setting::LoopCloserSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::boolalpha;
  os << std::right << std::setw(20) << "Use LoopCloser: "       << std::left << std::setw(10) << setting.use_loop_closer_ << std::endl
     << std::right << std::setw(20) << "Loop Threshold: "       << std::left << std::setw(10) << setting.loop_threshold_;
  return os;
}

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

inline std::ostream& operator<<(std::ostream& os,
                                const Setting::ExtrinsicSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::right << std::setw(20) << "Camera.Num: "       << std::left << std::setw(10) << setting.camera_num_ << std::endl
     << std::right << std::setw(20) << "Tbc0: "             << std::endl << setting.Tbc0_.matrix3x4().array() << std::endl
     << std::right << std::setw(20) << "Tbc1: "             << std::endl << setting.Tbc1_.matrix3x4().array();
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os, const Setting::FeatureAndTrackerSetting& setting) {
  os << std::right << std::setw(24) << GREEN << setting.name_ << TAIL << std::endl;
  os << std::boolalpha;
  os << std::right << std::setw(20) << "Max.Feature.Num: " << std::left << std::setw(10) <<  setting.max_feature_num_ << std::endl
     << std::right << std::setw(20) << "Max.Feature.Dis: " << std::left << std::setw(10) <<  setting.max_feature_dis_ << std::endl
     << std::right << std::setw(20) << "Quality.Level: "   << std::left << std::setw(10) <<  setting.feature_quality_level_ << std::endl
     << std::right << std::setw(20) << "Use.CLAHE: "       << std::left << std::setw(10) <<  setting.use_clahe_ << std::endl
     << std::right << std::setw(20) << "CLAHE.Val: "       << std::left << std::setw(10) <<  setting.clahe_val_ << std::endl
     << std::right << std::setw(20) << "KLT.Pyra.Level: "  << std::left << std::setw(10) <<  setting.klt_pyr_level_ << std::endl
     << std::right << std::setw(20) << "KLT.Win.Size: "    << std::left << std::setw(10) <<  setting.klt_win_size_ << std::endl
     << std::right << std::setw(20) << "KLT.Use.Prior: "   << std::left << std::setw(10) <<  setting.klt_use_prior_ << std::endl
     << std::right << std::setw(20) << "Use.Brief: "       << std::left << std::setw(10) <<  setting.use_brief_ << std::endl
     << std::right << std::setw(20) << "Track.KF2Frame: "  << std::left << std::setw(10) <<  setting.track_keyframetoframe_ << std::endl
     << std::right << std::setw(20) << "KLT.Max.Iter: "    << std::left << std::setw(10) <<  setting.klt_max_iter_ << std::endl
     << std::right << std::setw(20) << "KLT.Max.Px.Prec: " << std::left << std::setw(10) <<  setting.klt_max_px_precision_ << std::endl
     << std::right << std::setw(20) << "KLT.Max.FB.Dis: "  << std::left << std::setw(10) <<  setting.klt_max_fb_dist_ << std::endl
     << std::right << std::setw(20) << "KLT.Err: "         << std::left << std::setw(10) <<  setting.klt_err_;
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

inline std::ostream& operator<<(std::ostream& os,
                                const Setting& setting) {
  os << BLUE << "-------------[Params Start]----------" << TAIL << std::endl;
  os << setting.cam_setting_ << "\n";
  os << setting.imu_setting_ << "\n";
  os << setting.slam_setting_ << "\n";
  os << setting.feat_tracker_setting_ << "\n";
  os << setting.extrinsic_setting_ << "\n";
  os << setting.loop_setting_ << "\n";
  os << BLUE << "-------------[Params End]------------" << TAIL << std::endl;
  return os;
}

// clang-format on
}  // namespace viohw

#endif  // VIO_HELLO_WORLD_SETTING_HPP
