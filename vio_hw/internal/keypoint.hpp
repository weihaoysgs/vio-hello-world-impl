#ifndef VIO_HELLO_WORLD_KEYPOINT_HPP
#define VIO_HELLO_WORLD_KEYPOINT_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace viohw {
struct Keypoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int lmid_;

  cv::Point2f px_;      // pixel coordinate
  cv::Point2f unpx_;    // un-distortion pixel coordinate
  Eigen::Vector3d bv_;  // normalized plane coordinate

  int scale_;
  float angle_;
  cv::Mat desc_;

  bool is3d_;

  bool is_stereo_;
  cv::Point2f rpx_;
  cv::Point2f runpx_;
  Eigen::Vector3d rbv_;

  bool is_re_tracked_;

  Keypoint()
      : lmid_(-1), scale_(0), angle_(-1.), is3d_(false), is_stereo_(false), is_re_tracked_(false) {}

  // For using kps in ordered containers
  bool operator<(const Keypoint &kp) const { return lmid_ < kp.lmid_; }
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_KEYPOINT_HPP
