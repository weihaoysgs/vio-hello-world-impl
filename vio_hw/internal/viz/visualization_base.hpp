#ifndef VIO_HELLO_WORLD_VISUALIZATION_BASE_HPP
#define VIO_HELLO_WORLD_VISUALIZATION_BASE_HPP

#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace viohw {

class VisualizationBase
{
public:
  enum VisualizationPluginType
  {
    RVIZ,
    PANGOLIN,
  };

  struct VisualizationOption
  {
    VisualizationPluginType viz_type_;
  };

  virtual ~VisualizationBase() = default;

  virtual bool showTrackerResultImage( const cv::Mat &img ) = 0;

  virtual bool addTrajectory( const Eigen::Matrix3d &Q, const Eigen::Vector3d &t ) = 0;

  virtual bool addKFTrajectory( const Eigen::Matrix3d &Q, const Eigen::Vector3d &t ) = 0;

  virtual bool addPoint( const Eigen::Vector3d &t, const Eigen::Vector3d &color ) = 0;

  virtual bool showPoint() = 0;

  virtual bool showTrajectory() = 0;

  virtual bool showKFTrajectory() = 0;

  virtual bool clearKFTraj() {
    LOG( WARNING ) << "clearKFTraj tobe impl";
    return false;
  };

  virtual bool showLoopResultImage( const cv::Mat &img ) {
    LOG( WARNING ) << "showLoopResultImage tobe impl";
    return false;
  };

  virtual bool showRealtimeIMUState() {
    LOG( WARNING ) << "showLoopResultImage tobe impl";
    return false;
  };

  static std::shared_ptr<VisualizationBase> Create( const VisualizationOption &options );
};

typedef std::shared_ptr<VisualizationBase> VisualizationBasePtr;
typedef std::shared_ptr<const VisualizationBase> VisualizationBaseConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_VISUALIZATION_BASE_HPP
