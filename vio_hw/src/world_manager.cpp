#include "vio_hw/internal/world_manager.hpp"

#include "vio_hw/internal/feat/good_feature_impl.hpp"
#include "vio_hw/internal/feat/orb_slam_impl.hpp"
#include "vio_hw/internal/feat/superpoint_impl.hpp"
#include "vio_hw/internal/tracker/lightglue_impl.hpp"
#include "vio_hw/internal/tracker/optical_flow_impl.hpp"

namespace viohw {

WorldManager::WorldManager( std::shared_ptr<Setting>& setting ) : params_( setting ) {
  setupCalibration();

  // create feature extractor
  GenerateFeatureExtractorBase();

  // create feature tracker
  GenerateFeatureTrackerMatcherBase();

  // create IMU database
  imu_database_.reset( new backend::IMU::IMUDataBase( params_->imu_setting_.imu_FPS_,
                                                      params_->cam_setting_.camera_FPS_ ) );

  // create visualization
  VisualizationBase::VisualizationOption viz_option{ VisualizationBase::RVIZ };
  viz_ = VisualizationBase::Create( viz_option );

  system_state_.reset( new SystemState );

  // TODO: for [ncellsize] param
  // create current frame
  if ( !params_->slam_setting_.stereo_mode_ ) {
    current_frame_.reset( new Frame( calib_model_left_, 35 ) );
  } else {
    current_frame_.reset( new Frame( calib_model_left_, calib_model_right_, 35 ) );
  }

  // create map manager
  map_manager_.reset(
      new MapManager( params_, current_frame_, feature_extractor_, tracker_for_mapping_ ) );

  // create visual frontend
  visual_frontend_.reset(
      new VisualFrontEnd( params_, current_frame_, map_manager_, tracker_, viz_, system_state_ ) );

  // create optimization
  optimization_.reset( new Optimization( params_, map_manager_ ) );

  // create loop closer
  loop_closer_.reset( new LoopCloser( params_, map_manager_, optimization_ ) );

  // create estimator thread
  estimator_.reset( new Estimator( params_, map_manager_, optimization_, system_state_ ) );

  // create mapping thread, and mapping will create sub thread for Estimator and LoopClosing
  mapping_.reset( new Mapping( params_, map_manager_, current_frame_, loop_closer_, estimator_ ) );
}

void WorldManager::run() {
  cv::Mat img_left, img_right;
  double cur_time = 0, last_time = 0;
  std::vector<backend::IMU::Point> imus;
  bool use_imu = params_->slam_setting_.use_imu_;
  Sophus::SE3d Tbc = params_->extrinsic_setting_.Tbc0_;
  auto acc_n = static_cast<float>( params_->imu_setting_.acc_n_ );
  auto acc_w = static_cast<float>( params_->imu_setting_.acc_w_ );
  auto gyr_n = static_cast<float>( params_->imu_setting_.gyr_n_ );
  auto gyr_w = static_cast<float>( params_->imu_setting_.gyr_w_ );

  // TODO Only for test
  Eigen::Vector3d MH_05_gt_bias_acc( -0.020544, 0.124837, 0.061800 );
  Eigen::Vector3d MH_05_gt_bias_gyr( -0.001806, 0.020940, 0.076870 );

  backend::IMU::Calib calib( Tbc.cast<float>(), gyr_n, acc_n, gyr_w, acc_w );
  backend::IMU::Bias bias( MH_05_gt_bias_acc, MH_05_gt_bias_gyr );

  TimeStamp current_process_time, last_process_time;
  long process_fps = 0;

  while ( true ) {
    if ( getNewImage( img_left, img_right, cur_time ) ) {
      current_process_time = std::chrono::high_resolution_clock ::now();
      if ( use_imu ) {
        // reset preintegration from last frame, using last frame bias
        current_frame_->imu_state_.resetPreIntegrationFromLastFrame( bias, calib );
        // get interval imu measurement form last_time to cur_time
        auto status = imu_database_->GetIntervalIMUMeasurement( last_time, cur_time, imus );

        if ( status != backend::IMUMeasureStatus::SUCCESS_GET_IMU_DATA ) {
          // if not success get imu measure, set imu measure failed
          LOG( WARNING ) << backend::IMUMeasureStatusToString( status )
                         << ", size : " << imus.size();
          current_frame_->imu_state_.setIMUMeasureAvailable( false );
        } else {
          // set imu measure available success and integrate imu measures
          current_frame_->imu_state_.setIMUMeasureAvailable( true );
          PreIntegrateIMU( imus, last_time, cur_time );
        }
        imu_database_->EraseOlderIMUMeasure( last_time );

        // using imu acc init attitude
        if ( !is_init_imu_pose_ ) {
          Eigen::Matrix3d R;
          if ( !imu_database_->IMUAttitudeInit( imus, R ) ) {
            continue;
          }
          is_init_imu_pose_ = true;
          Sophus::SE3d Twb0( R, Eigen::Vector3d::Zero() );
          Sophus::SE3d Twc( Twb0 * Tbc );
          current_frame_->SetTwc( Twc );
        }
      }
      frame_id_++;
      current_frame_->updateFrame( frame_id_, cur_time );

      last_time = cur_time;

      // tracker frame to frame
      bool is_kf = visual_frontend_->VisualTracking( img_left, cur_time );

      // visualization tracking result and trajectory
      Visualization();

      // create keyframe and visualization kf trajectory
      if ( is_kf ) {
        Keyframe kf( current_frame_->kfid_, img_left, img_right,
                     visual_frontend_->GetCurrentFramePyramid() );

        // add keyframe to mapping thread
        mapping_->AddNewKf( kf );

        if ( !kf_viz_is_on_ ) {
          std::thread kf_viz_thread( &WorldManager::VisualizationKFTraj, this );
          kf_viz_thread.detach();
        }
      }

      process_fps = 1000 / std::chrono::duration_cast<std::chrono::milliseconds>(
                               current_process_time - last_process_time )
                               .count();
      last_process_time = current_process_time;
    }
    std::chrono::milliseconds dura( 1 );
    std::this_thread::sleep_for( dura );
  }
}

bool WorldManager::GenerateFeatureTrackerMatcherBase() {
  TrackerBase::TrackerOption tracker_option{ .tracker_type_ =
                                                 params_->feat_tracker_setting_.tracker_type_ };

  tracker_option.opticalFlowImplConfig.reset( new OpticalFlowImplConfig(
      params_->feat_tracker_setting_.klt_win_size_, params_->feat_tracker_setting_.klt_pyr_level_,
      params_->feat_tracker_setting_.klt_err_, params_->feat_tracker_setting_.klt_max_fb_dist_ ) );

  tracker_option.lightGlueImplConfig.reset( new LightGlueImplConfig );
  tracker_option.lightGlueImplConfig->max_kps_num_ =
      params_->feat_tracker_setting_.max_feature_num_;
  tracker_option.lightGlueImplConfig->config_file_path_ =
      params_->config_file_path_setting_.dfm_config_path_;

  tracker_ = TrackerBase::Create( tracker_option );
  tracker_for_mapping_ = TrackerBase::Create( tracker_option );

  return true;
}

bool WorldManager::GenerateFeatureExtractorBase() {
  if ( params_->feat_tracker_setting_.tracker_type_ == TrackerBase::LIGHT_GLUE &&
       params_->feat_tracker_setting_.feature_type_ != FeatureBase::SUPER_POINT ) {
    LOG( FATAL ) << "Enable LIGHTGLUE Must Enable SuperPoint";
  }

  FeatureBase::FeatureExtractorOptions feature_options{
      .feature_type_ = params_->feat_tracker_setting_.feature_type_ };

  feature_options.orbslamExtractorConfig.reset( new ORBSLAMExtractorConfig );
  feature_options.orbslamExtractorConfig->iniThFAST_ = 20;
  feature_options.orbslamExtractorConfig->minThFAST_ = 7;
  feature_options.orbslamExtractorConfig->level_ = 8;
  feature_options.orbslamExtractorConfig->scale_factor_ = 1.2;
  feature_options.orbslamExtractorConfig->max_kps_ =
      params_->feat_tracker_setting_.max_feature_num_;

  feature_options.goodFeature2TrackerConfig.reset( new GoodFeature2TrackerConfig );
  feature_options.goodFeature2TrackerConfig->kps_min_distance_ =
      params_->feat_tracker_setting_.max_feature_dis_;
  feature_options.goodFeature2TrackerConfig->kps_quality_level_ =
      params_->feat_tracker_setting_.feature_quality_level_;
  feature_options.goodFeature2TrackerConfig->max_kps_num_ =
      params_->feat_tracker_setting_.max_feature_num_;

  feature_options.superPointExtractorConfig.reset( new SuperPointExtractorConfig );
  feature_options.superPointExtractorConfig->config_file_path_ =
      params_->config_file_path_setting_.dfm_config_path_;
  feature_options.superPointExtractorConfig->max_kps_ =
      params_->feat_tracker_setting_.max_feature_num_;

  feature_extractor_ = FeatureBase::Create( feature_options );

  return true;
}

void WorldManager::addNewMonoImage( const double time, cv::Mat& im0 ) {
  std::lock_guard<std::mutex> lock( img_mutex_ );
  img_left_queen_.push( im0 );
  img_time_queen_.push( time );

  is_new_img_available_ = true;
}

void WorldManager::addNewStereoImages( const double time, cv::Mat& im0, cv::Mat& im1 ) {
  std::lock_guard<std::mutex> lock( img_mutex_ );
  img_left_queen_.push( im0 );
  img_right_queen_.push( im1 );
  img_time_queen_.push( time );

  is_new_img_available_ = true;
}

bool WorldManager::getNewImage( cv::Mat& iml, cv::Mat& imr, double& time ) {
  std::lock_guard<std::mutex> lock( img_mutex_ );

  if ( !is_new_img_available_ ) {
    return false;
  }
  int k = 0;

  do {
    k++;

    iml = img_left_queen_.front();
    img_left_queen_.pop();

    time = img_time_queen_.front();
    img_time_queen_.pop();

    if ( params_->slam_setting_.stereo_mode_ ) {
      imr = img_right_queen_.front();
      img_right_queen_.pop();
    }

    // if not force realtime, will process every frame
    if ( !params_->slam_setting_.force_realtime_ ) {
      break;
    }

  } while ( !img_left_queen_.empty() );

  if ( k > 1 ) {
    LOG( WARNING ) << " SLAM is late! Skipped " << k - 1 << " frames...\n";
  }

  if ( img_left_queen_.empty() ) {
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
    calib_model_right_->setupExtrinsic( params_->extrinsic_setting_.T_left_right_ );
  }
}

bool WorldManager::VisualizationImage() {
  // direct get tracker image in frontend thread is thread-safe
  bool s1 = viz_->showTrackerResultImage( visual_frontend_->GetTrackResultImage() );
  bool s2 = viz_->showLoopResultImage( loop_closer_->GetLoopMatcherResult() );
  return s1 && s2;
}

void WorldManager::VisualizationKFTraj() {
  kf_viz_is_on_ = true;

  viz_->clearKFTraj();
  for ( int i = 0; i < current_frame_->kfid_; i++ ) {
    auto kf = map_manager_->GetKeyframe( i );
    if ( kf == nullptr ) continue;
    Sophus::SE3d kf_pose = kf->GetTwc();
    viz_->addKFTrajectory( kf_pose.rotationMatrix(), kf_pose.translation() );
  }

  viz_->showKFTrajectory();

  kf_viz_is_on_ = false;
}

void WorldManager::SaveKFTrajectoryTUM( const std::string path ) {
  std::ofstream fout( path, std::ofstream::out );
  for ( int i = 0; i < current_frame_->kfid_; i++ ) {
    auto kf = map_manager_->GetKeyframe( i );
    if ( kf == nullptr ) continue;
    Sophus::SE3d kf_pose = kf->GetTwc();
    Eigen::Vector3d p = kf_pose.translation();
    Eigen::Quaternion q = kf_pose.unit_quaternion();
    double time = kf->img_time_;
    fout << std::setprecision( 20 ) << time << std::setprecision( 4 ) << " " << p.x() << " "
         << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
         << std::endl;
  }
  LOG( INFO ) << "KF Traj save at: " << path;
}

void WorldManager::InsertIMUMeasure( backend::IMU::Point& data ) {
  imu_database_->InsertMeasure( data );
}

void WorldManager::PreIntegrateIMU( vector<backend::IMU::Point>& imus, double last_image_time,
                                    double curr_image_time ) {
  imu_database_->PreIntegrateIMU( imus, last_image_time, curr_image_time,
                                  current_frame_->imu_state_.preintegrated_from_last_frame_,
                                  current_frame_->imu_state_.preintegrated_from_last_kf_ );
}

void WorldManager::Visualization() {
  viz_->addTrajectory( current_frame_->GetTwc().rotationMatrix(),
                       current_frame_->GetTwc().translation() );
  viz_->showTrajectory();
  VisualizationImage();
}

}  // namespace viohw