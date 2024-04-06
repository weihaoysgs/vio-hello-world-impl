#include "vio_hw/internal/viz/rviz_visualization.hpp"

namespace viohw {

RvizVisualization::RvizVisualization() : camera_pose_visual_( 1, 0, 0, 1 ) {
  LOG( INFO ) << "Create RvizVisualization";
  ros::NodeHandle n( "~" );
  camera_pose_visual_pub_ = n.advertise<visualization_msgs::MarkerArray>( "cam_pose_visual", 1000 );
  pub_img_tracker_result_ = n.advertise<sensor_msgs::Image>( "image_track", 1000 );
  pub_loop_img_result_ = n.advertise<sensor_msgs::Image>( "loop_image", 1000 );
  pub_vo_traj_ = n.advertise<nav_msgs::Path>( "vo_traj", 100 );
  pub_kfs_traj_ = n.advertise<nav_msgs::Path>( "kf_traj", 100 );
  vo_traj_path_.header.stamp = ros::Time::now();
  vo_traj_path_.header.frame_id = world_frame_;
  kf_traj_path_.header.stamp = ros::Time::now();
  kf_traj_path_.header.frame_id = world_frame_;
}

bool RvizVisualization::addTrajectory( const Eigen::Matrix3d& R, const Eigen::Vector3d& t ) {
  if ( pub_vo_traj_.getNumSubscribers() == 0 ) {
    return false;
  }
  geometry_msgs::PoseStamped pose;
  pose.header.stamp = ros::Time::now();
  Eigen::Quaterniond Q( R );
  pose.pose.orientation.x = Q.x();
  pose.pose.orientation.y = Q.y();
  pose.pose.orientation.z = Q.z();
  pose.pose.orientation.w = Q.w();
  pose.pose.position.x = t.x();
  pose.pose.position.y = t.y();
  pose.pose.position.z = t.z();
  vo_traj_path_.poses.push_back( pose );

  // Publish camera visual
  camera_pose_visual_.reset();
  camera_pose_visual_.add_pose( t, Q );
  camera_pose_visual_.setImageBoundaryColor( 1, 0, 0 );
  camera_pose_visual_.setOpticalCenterConnectorColor( 1, 0, 0 );
  camera_pose_visual_.publish_by( camera_pose_visual_pub_, vo_traj_path_.header );

  return true;
}

bool RvizVisualization::showTrackerResultImage( const cv::Mat& img ) {
  if ( pub_img_tracker_result_.getNumSubscribers() == 0 ) {
    return false;
  }

  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time::now();
  sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage( header, "rgb8", img ).toImageMsg();
  pub_img_tracker_result_.publish( imgTrackMsg );
  return true;
}

// TODO: optimization
bool RvizVisualization::addKFTrajectory( const Eigen::Matrix3d& R, const Eigen::Vector3d& t ) {
  if ( pub_kfs_traj_.getNumSubscribers() == 0 ) {
    return false;
  }
  geometry_msgs::PoseStamped pose;
  pose.header.stamp = ros::Time::now();
  Eigen::Quaterniond Q( R );
  pose.pose.orientation.x = Q.x();
  pose.pose.orientation.y = Q.y();
  pose.pose.orientation.z = Q.z();
  pose.pose.orientation.w = Q.w();
  pose.pose.position.x = t.x();
  pose.pose.position.y = t.y();
  pose.pose.position.z = t.z();
  kf_traj_path_.poses.push_back( pose );

  return true;
}

bool RvizVisualization::addPoint( const Eigen::Vector3d& t, const Eigen::Vector3d& color ) {
  std::cout << "rviz addPoint\n";
  return false;
}

bool RvizVisualization::showLoopResultImage( const cv::Mat& img ) {
  if ( pub_loop_img_result_.getNumSubscribers() == 0 ) {
    return false;
  }
  if ( img.empty() ) {
    return false;
  }

  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time::now();
  sensor_msgs::ImagePtr loop_img = cv_bridge::CvImage( header, "rgb8", img ).toImageMsg();
  pub_loop_img_result_.publish( loop_img );
  return true;
}

bool RvizVisualization::clearKFTraj() {
  kf_traj_path_.poses.clear();
  return true;
}

bool RvizVisualization::showPoint() { return false; }

bool RvizVisualization::showTrajectory() {
  if ( pub_vo_traj_.getNumSubscribers() == 0 ) {
    return false;
  }
  pub_vo_traj_.publish( vo_traj_path_ );
  return true;
}

bool RvizVisualization::showKFTrajectory() {
  if ( pub_kfs_traj_.getNumSubscribers() == 0 ) {
    return false;
  }
  pub_kfs_traj_.publish( kf_traj_path_ );
  return true;
}

}  // namespace viohw
