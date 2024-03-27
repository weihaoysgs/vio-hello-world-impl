#include "vio_hw/internal/viz/rviz_visualization.hpp"

namespace viohw {

bool RvizVisualization::addTrajectory(const Eigen::Matrix3d& Q,
                                      const Eigen::Vector3d& t) {
  std::cout << "rviz addTrajectory\n";
  return true;
}

bool RvizVisualization::showTrackerResultImage(const cv::Mat& img) {
  std::cout << "rviz showTrackerResultImage\n";
  return false;
}

bool RvizVisualization::addKFTrajectory(const Eigen::Matrix3d& Q,
                                        const Eigen::Vector3d& t) {
  std::cout << "rviz addKFTrajectory\n";
  return false;
}

bool RvizVisualization::addPoint(const Eigen::Vector3d& t,
                                 const Eigen::Vector3d& color) {
  std::cout << "rviz addPoint\n";
  return false;
}

}  // namespace viohw
