#ifndef VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP
#define VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>

#include "vio_hw/internal/viz/camera_visualizer.hpp"
#include "vio_hw/internal/viz/visualization_base.hpp"

namespace viohw {

class RvizVisualization : public VisualizationBase
{
public:
  explicit RvizVisualization();

  bool showTrackerResultImage( const cv::Mat &img ) override;

  bool addTrajectory( const Eigen::Matrix3d &Q, const Eigen::Vector3d &t ) override;

  bool addKFTrajectory( const Eigen::Matrix3d &Q, const Eigen::Vector3d &t ) override;

  bool addPoint( const Eigen::Vector3d &t, const Eigen::Vector3d &color ) override;

  bool showLoopResultImage( const cv::Mat &img ) override;

  bool clearKFTraj() override;

  bool showPoint() override;

  bool showTrajectory() override;

  bool showKFTrajectory() override;

private:
  CameraPoseVisualization camera_pose_visual_;
  ros::Publisher camera_pose_visual_pub_;
  ros::Publisher pub_kfs_traj_, pub_vo_traj_;
  ros::Publisher pub_img_tracker_result_, pub_loop_img_result_;
  nav_msgs::Path vo_traj_path_, kf_traj_path_;
  const std::string world_frame_ = "map";
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP
