#ifndef VIO_HELLO_WORLD_PANGOLIN_VISUALIZATION_HPP
#define VIO_HELLO_WORLD_PANGOLIN_VISUALIZATION_HPP

#include "vio_hw/internal/viz/visualization_base.hpp"

namespace viohw {

class PangolinVisualization : public VisualizationBase
{
 public:
  PangolinVisualization() = default;

  void render();

  bool showTrackerResultImage(const cv::Mat &img) override;

  bool addTrajectory(const Eigen::Matrix3d &Q,
                     const Eigen::Vector3d &t) override;

  bool addKFTrajectory(const Eigen::Matrix3d &Q,
                       const Eigen::Vector3d &t) override;

  bool addPoint(const Eigen::Vector3d &t,
                const Eigen::Vector3d &color) override;
};

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_PANGOLIN_VISUALIZATION_HPP
