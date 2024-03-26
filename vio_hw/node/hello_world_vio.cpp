// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

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
#include "vio_hw/vio/world_manager.hpp"

DEFINE_string(config_file_path, "../vio_hw/params/euroc_stereo_imu.yaml",
              "config file path");

class SensorManager
{
 public:
  explicit SensorManager(viohw::WorldManager *slam_manager)
      : slam_(slam_manager), nh_("~") {
    LOG(INFO) << "Sensor Manager is create.";

    sub_left_img_ = nh_.subscribe(slam_->getParams()->topic_left_right_[0], 2,
                                  &SensorManager::subLeftImage, this);
    sub_right_img_ = nh_.subscribe(slam_->getParams()->topic_left_right_[1], 2,
                                   &SensorManager::subRightImage, this);
    sub_imu_ = nh_.subscribe(slam_->getParams()->imu_topic_, 10,
                             &SensorManager::subIMU, this);
    std::printf("sub image0 topic :%s, img topic %s, imu topic %s",
                slam_->getParams()->topic_left_right_[0].c_str(),
                slam_->getParams()->topic_left_right_[1].c_str(),
                slam_->getParams()->imu_topic_.c_str());
  };

  ~SensorManager() = default;

  void subLeftImage(const sensor_msgs::ImageConstPtr &image) {
    std::lock_guard<std::mutex> lock(img_mutex_);
    img0_buf_.push(image);
  }

  void subRightImage(const sensor_msgs::ImageConstPtr &image) {
    std::lock_guard<std::mutex> lock(img_mutex_);
    img1_buf_.push(image);
  }

  void subIMU(const sensor_msgs::ImuConstPtr &imu) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_buf_.push(imu);
  }

  void syncProcess() {
    LOG(INFO) << "Start the measurement reader thread";
    while (true) {
      std::printf(
          "sync process, img0 size: %ld, img1 size: %ld, imu size: %ld\n",
          img0_buf_.size(), img1_buf_.size(), imu_buf_.size());
      std::chrono::milliseconds dura(100);
      std::this_thread::sleep_for(dura);
    }
  }

 private:
  std::queue<sensor_msgs::ImageConstPtr> img0_buf_;
  std::queue<sensor_msgs::ImageConstPtr> img1_buf_;
  std::queue<sensor_msgs::ImuConstPtr> imu_buf_;
  std::mutex img_mutex_, imu_mutex_;
  viohw::WorldManager *slam_;
  ros::Subscriber sub_left_img_, sub_right_img_, sub_imu_;
  ros::NodeHandle nh_;
};

int main(int argc, char **argv) {
  google::InitGoogleLogging("hello_world_vio");
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  ros::init(argc, argv, "hello_world_vio");

  auto params = std::make_shared<viohw::Setting>(FLAGS_config_file_path);
  viohw::WorldManager world_manager(params);
  // Start the SLAM thread
  std::thread estimate_thread(&viohw::WorldManager::run, &world_manager);

  SensorManager sensor_manager(&world_manager);
  // Start the sensor measurement grab thread
  std::thread sensor_grab_thread(&SensorManager::syncProcess, &sensor_manager);

  ros::spin();
  return 0;
}