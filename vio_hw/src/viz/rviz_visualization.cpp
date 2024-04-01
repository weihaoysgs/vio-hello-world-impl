#include "vio_hw/internal/viz/rviz_visualization.hpp"

namespace viohw {

RvizVisualization::RvizVisualization() : camera_pose_visual_(1, 0, 0, 1) {
  LOG(INFO) << "Create RvizVisualization";
  ros::NodeHandle n("~");
  pub_img_tracker_result_ = n.advertise<sensor_msgs::Image>("image_track", 1000);
  pub_vo_traj_ = n.advertise<nav_msgs::Path>("vo_traj", 100);
  vo_traj_path_.header.stamp = ros::Time::now();
  vo_traj_path_.header.frame_id = world_frame_;
}

bool RvizVisualization::addTrajectory(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  geometry_msgs::PoseStamped pose;
  pose.header.stamp = ros::Time::now();
  Eigen::Quaterniond Q(R);
  pose.pose.orientation.x = Q.x();
  pose.pose.orientation.y = Q.y();
  pose.pose.orientation.z = Q.z();
  pose.pose.orientation.w = Q.w();
  pose.pose.position.x = t.x();
  pose.pose.position.y = t.y();
  pose.pose.position.z = t.z();
  vo_traj_path_.poses.push_back(pose);
  pub_vo_traj_.publish(vo_traj_path_);
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

}  // namespace viohw
