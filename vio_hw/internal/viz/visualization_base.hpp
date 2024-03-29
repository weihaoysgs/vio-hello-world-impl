#ifndef VIO_HELLO_WORLD_VISUALIZATION_BASE_HPP
#define VIO_HELLO_WORLD_VISUALIZATION_BASE_HPP

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

  virtual bool showTrackerResultImage(const cv::Mat &img) = 0;

  virtual bool addTrajectory(const Eigen::Matrix3d &Q,
                             const Eigen::Vector3d &t) = 0;

  virtual bool addKFTrajectory(const Eigen::Matrix3d &Q,
                               const Eigen::Vector3d &t) = 0;

  virtual bool addPoint(const Eigen::Vector3d &t,
                        const Eigen::Vector3d &color) = 0;

  static std::shared_ptr<VisualizationBase> Create(
      const VisualizationOption &options);

};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_VISUALIZATION_BASE_HPP
