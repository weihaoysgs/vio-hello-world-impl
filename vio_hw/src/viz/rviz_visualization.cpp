#include "vio_hw/internal/viz/rviz_visualization.hpp"

namespace viohw {

bool RvizVisualization::addTrajectory(const Eigen::Matrix3d& Q, const Eigen::Vector3d& t) {
  std::cout << "rviz addTrajectory\n";
  return true;
}

bool RvizVisualization::showTrackerResultImage(const cv::Mat& img) {
  if (pub_img_tracker_result_.getNumSubscribers() == 0) {
    return false;
  }

  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time::now();
  sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "rgb8", img).toImageMsg();
  pub_img_tracker_result_.publish(imgTrackMsg);
  return true;
}

bool RvizVisualization::addKFTrajectory(const Eigen::Matrix3d& Q, const Eigen::Vector3d& t) {
  std::cout << "rviz addKFTrajectory\n";
  return false;
}

bool RvizVisualization::addPoint(const Eigen::Vector3d& t, const Eigen::Vector3d& color) {
  std::cout << "rviz addPoint\n";
  return false;
}

RvizVisualization::RvizVisualization() : camera_pose_visual_(1, 0, 0, 1) {
  LOG(INFO) << "Create RvizVisualization";
  ros::NodeHandle n("~");
  pub_img_tracker_result_ = n.advertise<sensor_msgs::Image>("image_track", 1000);
}

}  // namespace viohw
