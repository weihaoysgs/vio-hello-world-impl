#ifndef VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP
#define VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP

#include "vio_hw/internal/viz/visualization_base.hpp"

namespace viohw {

class RvizVisualization : public VisualizationBase
{
 public:
  RvizVisualization() = default;

  bool showTrackerResultImage(const cv::Mat &img) override;

  bool addTrajectory(const Eigen::Matrix3d &Q,
                     const Eigen::Vector3d &t) override;

  bool addKFTrajectory(const Eigen::Matrix3d &Q,
                       const Eigen::Vector3d &t) override;

  bool addPoint(const Eigen::Vector3d &t,
                const Eigen::Vector3d &color) override;
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_RVIZ_VISUALIZATION_HPP
