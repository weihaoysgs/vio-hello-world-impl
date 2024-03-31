#ifndef VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP
#define VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP

#include "vio_hw/internal/viz/visualization_base.hpp"
#include "vio_hw/internal/viz/camera_visualizer.hpp"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <glog/logging.h>
#include <cv_bridge/cv_bridge.h>

namespace viohw {

class RvizVisualization : public VisualizationBase
{
 public:
  explicit RvizVisualization();

  bool showTrackerResultImage(const cv::Mat &img) override;

  bool addTrajectory(const Eigen::Matrix3d &Q,
                     const Eigen::Vector3d &t) override;

  bool addKFTrajectory(const Eigen::Matrix3d &Q,
                       const Eigen::Vector3d &t) override;

  bool addPoint(const Eigen::Vector3d &t,
                const Eigen::Vector3d &color) override;

 private:
  CameraPoseVisualization camera_pose_visual_;
  ros::Publisher pub_kfs_traj_, pub_vo_traj_;
  ros::Publisher pub_img_tracker_result_;
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP
