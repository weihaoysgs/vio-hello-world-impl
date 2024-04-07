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

#include "feat/harris/harris.h"
#include "tceres/problem.h"
#include "tceres/solver.h"
#include "vio_hw/internal/world_manager.hpp"

DEFINE_string( config_file_path, "../vio_hw/params/kitti/kitti_00-02.yaml", "config file path" );
DEFINE_string( kitti_dataset_path,
               "/home/weihao/dataset/kitti/data_odometry_gray/dataset/sequences/00",
               "kitti dataset path" );
void LoadKittiImagesTimestamps( const std::string &str_path_to_sequence,
                                std::vector<std::string> &str_image_left_vec_path,
                                std::vector<std::string> &str_image_right_vec_path,
                                std::vector<double> &timestamps_vec );
class SensorManager
{
public:
  explicit SensorManager( viohw::WorldManager *slam_manager ) : slam_world_( slam_manager ) {
    LOG( INFO ) << "Sensor Manager is create.";
    LoadKittiImagesTimestamps( fLS::FLAGS_kitti_dataset_path, image_left_vec_path,
                               image_right_vec_path, vec_timestamp );
  };

  ~SensorManager() = default;

  void syncProcess() {
    LOG( INFO ) << "Start the measurement reader thread";

    const size_t num_images = image_left_vec_path.size();

    cv::Mat debug_img( 380, 1180, CV_8UC3, cv::Scalar( 0, 0, 0 ) );
    bool debug_stepbystep = false;
    cv::putText( debug_img, "Press Space Step By Step", cv::Point2i( 100, 100 ), 3, 2,
                 cv::Scalar( 0, 0, 255 ), 2 );

    for ( int ni = 0; ni < num_images; ni++ ) {
      if ( slam_world_->getParams()->slam_setting_.stereo_mode_ ) {
        std::lock_guard<std::mutex> lock( img_mutex_ );

        double timestamp = vec_timestamp[ni];
        cv::Mat image0 = cv::imread( image_left_vec_path[ni], cv::IMREAD_GRAYSCALE );
        cv::Mat image1 = cv::imread( image_right_vec_path[ni], cv::IMREAD_GRAYSCALE );

        assert( !image0.empty() && !image1.empty() );
        slam_world_->addNewStereoImages( timestamp, image0, image1 );

        if ( debug_stepbystep ) {
          cv::imshow( "image", debug_img );
          cv::waitKey( 0 );
        }
      }
      std::chrono::milliseconds dura( 20 );
      std::this_thread::sleep_for( dura );
    }
  }

private:
  std::queue<cv::Mat> img0_buf_;
  std::queue<cv::Mat> img1_buf_;
  std::mutex img_mutex_;
  std::vector<std::string> image_left_vec_path, image_right_vec_path;
  std::vector<double> vec_timestamp;
  viohw::WorldManager *slam_world_;
};

int main( int argc, char **argv ) {
  google::InitGoogleLogging( "hello_world_vio" );
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags( &argc, &argv, true );
  ros::init( argc, argv, "hello_world_vio" );

  auto params = std::make_shared<viohw::Setting>( FLAGS_config_file_path );
  std::cout << *params << std::endl;
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

inline void LoadKittiImagesTimestamps( const std::string &str_path_to_sequence,
                                       std::vector<std::string> &str_image_left_vec_path,
                                       std::vector<std::string> &str_image_right_vec_path,
                                       std::vector<double> &timestamps_vec ) {
  using namespace std;
  string strPathTimeFile = str_path_to_sequence + "/times.txt";

  std::ifstream fTimes( strPathTimeFile, ios::in | ios::app );

  if ( !fTimes.is_open() ) LOG( FATAL ) << "Open Failed";
  while ( !fTimes.eof() ) {
    string s;
    getline( fTimes, s );
    if ( !s.empty() ) {
      stringstream ss;
      ss << s;
      double t;
      ss >> t;
      timestamps_vec.push_back( t );
    }
  }

  string strPrefixLeft = str_path_to_sequence + "/image_0/";
  string strPrefixRight = str_path_to_sequence + "/image_1/";

  const size_t nTimes = timestamps_vec.size();
  str_image_left_vec_path.resize( nTimes );
  str_image_right_vec_path.resize( nTimes );

  for ( int i = 0; i < nTimes; i++ ) {
    stringstream ss;
    ss << setfill( '0' ) << setw( 6 ) << i;
    str_image_left_vec_path[i] = strPrefixLeft + ss.str() + ".png";
    str_image_right_vec_path[i] = strPrefixRight + ss.str() + ".png";
  }
  fTimes.close();
}
