#include "vio_hw/internal/viz/pangolin_visualization.hpp"

namespace viohw {

bool PangolinVisualization::showTrackerResultImage(const cv::Mat& img) {
  std::cout << "Pangolin showTrackerResultImage\n";
  return false;
}

bool PangolinVisualization::addTrajectory(const Eigen::Matrix3d& Q,
                                          const Eigen::Vector3d& t) {
  std::cout << "Pangolin addTrajectory\n";
  return false;
}

bool PangolinVisualization::addKFTrajectory(const Eigen::Matrix3d& Q,
                                            const Eigen::Vector3d& t) {
  std::cout << "Pangolin addKFTrajectory\n";
  return false;
}

bool PangolinVisualization::addPoint(const Eigen::Vector3d& t,
                                     const Eigen::Vector3d& color) {
  std::cout << "Pangolin addPoint\n";
  return false;
}

void PangolinVisualization::render() {}
bool PangolinVisualization::showPoint() { return false; }
bool PangolinVisualization::showTrajectory() { return false; }
bool PangolinVisualization::showKFTrajectory() { return false; }

}  // namespace viohw
