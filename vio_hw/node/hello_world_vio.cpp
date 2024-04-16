// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include <cv_bridge/cv_bridge.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>

#include <thread>

#include "backend/sensor_fusion/imu/imu_types.h"
#include "feat/harris/harris.h"
#include "tceres/problem.h"
#include "tceres/solver.h"
#include "vio_hw/internal/world_manager.hpp"

DEFINE_string( config_file_path, "../vio_hw/params/euroc/euroc_stereo_imu.yaml",
               "config file path" );

class SensorManager
{
public:
  explicit SensorManager( viohw::WorldManager *slam_manager )
      : slam_world_( slam_manager ), nh_( "~" ) {
    LOG( INFO ) << "Sensor Manager is create.";

    sub_left_img_ = nh_.subscribe( slam_world_->getParams()->cam_setting_.topic_left_right_[0], 1,
                                   &SensorManager::subLeftImage, this );
    if ( slam_world_->getParams()->slam_setting_.stereo_mode_ )
      sub_right_img_ = nh_.subscribe( slam_world_->getParams()->cam_setting_.topic_left_right_[1],
                                      1, &SensorManager::subRightImage, this );
    if ( slam_world_->getParams()->slam_setting_.use_imu_ )
      sub_imu_ = nh_.subscribe( slam_world_->getParams()->imu_setting_.imu_topic_, 10,
                                &SensorManager::subIMU, this );
  };

  ~SensorManager() = default;

  void subLeftImage( const sensor_msgs::ImageConstPtr &image ) {
    std::lock_guard<std::mutex> lock( img_mutex_ );
    img0_buf_.push( image );
  }

  void subRightImage( const sensor_msgs::ImageConstPtr &image ) {
    std::lock_guard<std::mutex> lock( img_mutex_ );
    img1_buf_.push( image );
  }

  void subIMU( const sensor_msgs::ImuConstPtr &imu ) {
    backend::IMU::Point data;
    data.a.x() = imu->linear_acceleration.x;
    data.a.y() = imu->linear_acceleration.y;
    data.a.z() = imu->linear_acceleration.z;
    data.w.x() = imu->angular_velocity.x;
    data.w.y() = imu->angular_velocity.y;
    data.w.z() = imu->angular_velocity.z;
    data.t = imu->header.stamp.toSec();
    slam_world_->InsertIMUMeasure( data );
  }

  cv::Mat getGrayImageFromMsg( const sensor_msgs::ImageConstPtr &img_msg ) {
    // Get and prepare images
    cv_bridge::CvImageConstPtr ptr;
    try {
      ptr = cv_bridge::toCvCopy( img_msg, sensor_msgs::image_encodings::MONO8 );
    }
    catch ( cv_bridge::Exception &e ) {
      ROS_ERROR( "\n\n\ncv_bridge exeception: %s\n\n\n", e.what() );
    }

    return ptr->image;
  }
  void syncProcess() {
    LOG( INFO ) << "Start the measurement reader thread";
    while ( true ) {
      if ( slam_world_->getParams()->slam_setting_.stereo_mode_ ) {
        cv::Mat image0, image1;
        std::lock_guard<std::mutex> lock( img_mutex_ );
        if ( !img0_buf_.empty() && !img1_buf_.empty() ) {
          double time0 = img0_buf_.front()->header.stamp.toSec();
          double time1 = img1_buf_.front()->header.stamp.toSec();
          // sync tolerance
          if ( time0 < time1 - 0.015 ) {
            img0_buf_.pop();
            LOG( WARNING ) << "Throw img0 -- Sync error : " << ( time0 - time1 );
          } else if ( time0 > time1 + 0.015 ) {
            img1_buf_.pop();
            LOG( WARNING ) << "Throw img1 -- Sync error : " << ( time0 - time1 );
          } else {
            image0 = getGrayImageFromMsg( img0_buf_.front() );
            image1 = getGrayImageFromMsg( img1_buf_.front() );
            img0_buf_.pop();
            img1_buf_.pop();

            if ( !image0.empty() && !image1.empty() ) {
              slam_world_->addNewStereoImages( time0, image0, image1 );
            }
          }
        }
      } else {
        cv::Mat image0;
        std::lock_guard<std::mutex> lock( img_mutex_ );

        if ( !img0_buf_.empty() ) {
          double time = img0_buf_.front()->header.stamp.toSec();
          image0 = getGrayImageFromMsg( img0_buf_.front() );
          img0_buf_.pop();

          if ( !image0.empty() ) {
            slam_world_->addNewMonoImage( time, image0 );
          }
        }
      }
      std::chrono::milliseconds dura( 1 );
      std::this_thread::sleep_for( dura );
    }
  }

private:
  std::queue<sensor_msgs::ImageConstPtr> img0_buf_;
  std::queue<sensor_msgs::ImageConstPtr> img1_buf_;

  std::mutex img_mutex_, imu_mutex_;
  viohw::WorldManager *slam_world_;
  ros::Subscriber sub_left_img_, sub_right_img_, sub_imu_;
  ros::NodeHandle nh_;
};

int main( int argc, char **argv ) {
  google::InitGoogleLogging( "hello_world_vio" );
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags( &argc, &argv, true );
  ros::init( argc, argv, "hello_world_vio" );

  auto params = std::make_shared<viohw::Setting>( FLAGS_config_file_path );
  std::cout << *params << std::endl;
  com::printHelloWorldVIO();

  viohw::WorldManager world_manager( params );
  // Start the SLAM thread
  std::thread estimate_thread( &viohw::WorldManager::run, &world_manager );

  SensorManager sensor_manager( &world_manager );
  // Start the sensor measurement grab thread
  std::thread sensor_grab_thread( &SensorManager::syncProcess, &sensor_manager );

  ros::spin();
  world_manager.SaveKFTrajectoryTUM( params->config_file_path_setting_.kf_traj_save_path_ );
  return 0;
}