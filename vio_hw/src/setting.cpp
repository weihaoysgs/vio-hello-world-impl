#include "vio_hw/internal/setting.hpp"

namespace viohw {

Setting::Setting( const std::string& config_file_path ) {
  cv::FileStorage fs( config_file_path, cv::FileStorage::READ );
  LOG_IF( FATAL, !fs.isOpened() ) << "Config file open failed, please check your confi file path.";
  Init( fs );
}

void Setting::Init( const cv::FileStorage& fs ) {
  cv::FileNode cameras = fs["Camera"];
  cv::FileNode imu = fs["IMU"];
  cv::FileNode slam = fs["SLAM"];
  cv::FileNode feat_tracker = fs["FeatureAndTracker"];
  cv::FileNode extrinsic = fs["Extrinsic"];
  cv::FileNode loop = fs["LoopCloser"];
  cv::FileNode configfile = fs["ConfigFilePath"];
  cv::FileNode backend = fs["Backend"];
  readCameraParams( cameras );
  readIMUParams( imu );
  readSLAMParams( slam );
  readFeatureTrackerParams( feat_tracker );
  readExtrinsicParams( extrinsic );
  readLoopCloserParams( loop );
  readConfigFilePathSetting( configfile );
  readBackendOptimizationSetting( backend );
}

void Setting::readCameraParams( const cv::FileNode& cameras ) {
  cameras["topic.left.right"] >> cam_setting_.topic_left_right_;
  cameras["model.left.right"] >> cam_setting_.model_left_right_;
  cameras["left.resolution"] >> cam_setting_.left_resolution_;
  cameras["right.resolution"] >> cam_setting_.right_resolution_;
  cameras["left.K"] >> cam_setting_.left_k_;
  cameras["right.K"] >> cam_setting_.right_k_;
  cameras["left.distortion.coefficient"] >> cam_setting_.left_dist_coeff_;
  cameras["right.distortion.coefficient"] >> cam_setting_.right_dist_coeff_;
}

void Setting::readIMUParams( const cv::FileNode& node ) {
  node["topic"] >> imu_setting_.imu_topic_;
  node["acc_n"] >> imu_setting_.acc_n_;
  node["acc_w"] >> imu_setting_.acc_w_;
  node["gyr_n"] >> imu_setting_.gyr_n_;
  node["gyr_w"] >> imu_setting_.gyr_w_;
}

void Setting::readSLAMParams( const cv::FileNode& node ) {
  node["stereo.mode"] >> slam_setting_.stereo_mode_;
  node["use.imu"] >> slam_setting_.use_imu_;
  node["force.realtime"] >> slam_setting_.force_realtime_;
  node["use.Pangolin"] >> slam_setting_.use_pangolin_;
  node["use.Rviz"] >> slam_setting_.use_rviz_;
}

void Setting::readFeatureTrackerParams( const cv::FileNode& node ) {
  node["max.kps.num"] >> feat_tracker_setting_.max_feature_num_;
  node["max.kps.distance"] >> feat_tracker_setting_.max_feature_dis_;
  node["kp.quality.level"] >> feat_tracker_setting_.feature_quality_level_;
  node["use.clahe"] >> feat_tracker_setting_.use_clahe_;
  node["clahe.val"] >> feat_tracker_setting_.clahe_val_;
  node["klt.use.prior"] >> feat_tracker_setting_.klt_use_prior_;
  node["track.keyframetoframe"] >> feat_tracker_setting_.track_keyframetoframe_;
  node["klt.win.size"] >> feat_tracker_setting_.klt_win_size_;
  node["klt.pyr.level"] >> feat_tracker_setting_.klt_pyr_level_;
  node["use.brief"] >> feat_tracker_setting_.use_brief_;

  node["klt.max.iter"] >> feat_tracker_setting_.klt_max_iter_;
  node["klt.max.px.precision"] >> feat_tracker_setting_.klt_max_px_precision_;
  node["klt.max.fb.dist"] >> feat_tracker_setting_.klt_max_fb_dist_;
  node["klt.err"] >> feat_tracker_setting_.klt_err_;
}

void Setting::readExtrinsicParams( const cv::FileNode& node ) {
  cv::Mat cvTbc0, cvTbc1;
  Eigen::Matrix4d Tbc0, Tbc1;

  node["body_T_cam0"] >> cvTbc0;
  node["body_T_cam1"] >> cvTbc1;

  cv::cv2eigen( cvTbc0, Tbc0 );
  cv::cv2eigen( cvTbc1, Tbc1 );

  auto normalizedRotation = []( Eigen::Matrix4d& Tic ) -> void {
    Eigen::Quaterniond unit_Q( Tic.block<3, 3>( 0, 0 ) );
    Eigen::Matrix3d Ric = unit_Q.normalized().toRotationMatrix();
    Eigen::Vector3d tic = Tic.block<3, 1>( 0, 3 );
    Tic = Sophus::SE3d( Ric, tic ).matrix();
  };

  normalizedRotation( Tbc0 );
  normalizedRotation( Tbc1 );

  extrinsic_setting_.T_left_right_ = Sophus::SE3d( Tbc0.inverse() * Tbc1 );
  extrinsic_setting_.Tc0c1_ = Sophus::SE3d( Tbc0.inverse() * Tbc1 );
  extrinsic_setting_.Tbc0_ = Sophus::SE3d( Tbc0 );
  extrinsic_setting_.Tbc1_ = Sophus::SE3d( Tbc1 );

  node["camera.num"] >> extrinsic_setting_.camera_num_;
}

void Setting::readLoopCloserParams( const cv::FileNode& node ) {
  node["use.loop"] >> loop_setting_.use_loop_closer_;
  node["loop.threshold"] >> loop_setting_.loop_threshold_;
}

void Setting::readConfigFilePathSetting( const cv::FileNode& node ) {
  node["vocabulary.path"] >> config_file_path_setting_.vocabulary_path_;
  node["kf.traj.save.path"] >> config_file_path_setting_.kf_traj_save_path_;
  node["bag.file.path"] >> config_file_path_setting_.bag_file_path_;
}

void Setting::readBackendOptimizationSetting( const cv::FileNode& node ) {
  node["opt.window.size"] >> backend_optimization_setting_.optimize_kf_num_;
  node["use.backend"] >> backend_optimization_setting_.open_backend_opt_;
}
}  // namespace viohw
