#ifndef VIO_HELLO_WORLD_TRIANGULATE_CV_HPP
#define VIO_HELLO_WORLD_TRIANGULATE_CV_HPP
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "sophus/se3.hpp"

namespace geometry {
inline Eigen::Vector3d OpencvTriangulate(const Sophus::SE3d &Tlr, const Eigen::Vector3d &bvl,
                                         const Eigen::Vector3d &bvr) {
  std::vector<cv::Point2f> lpt, rpt;
  lpt.push_back(cv::Point2f(bvl.x() / bvl.z(), bvl.y() / bvl.z()));
  rpt.push_back(cv::Point2f(bvr.x() / bvr.z(), bvr.y() / bvr.z()));

  cv::Matx34f P0 = cv::Matx34f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

  Sophus::SE3d Tcw = Tlr.inverse();
  Eigen::Matrix3d R = Tcw.rotationMatrix();
  Eigen::Vector3d t = Tcw.translation();

  cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0), R(1, 0), R(1, 1), R(1, 2), t(1),
                               R(2, 0), R(2, 1), R(2, 2), t(2));

  cv::Mat campt;
  cv::triangulatePoints(P0, P1, lpt, rpt, campt);

  if (campt.col(0).at<float>(3) != 1.) {
    campt.col(0) /= campt.col(0).at<float>(3);
  }

  Eigen::Vector3d pt(campt.col(0).at<float>(0), campt.col(0).at<float>(1),
                     campt.col(0).at<float>(2));

  return pt;
}

}  // namespace geometry

#endif  // VIO_HELLO_WORLD_TRIANGULATE_CV_HPP
