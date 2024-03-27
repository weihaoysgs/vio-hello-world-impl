#ifndef VIO_HELLO_WORLD_ORB_SLAM_IMPL_HPP
#define VIO_HELLO_WORLD_ORB_SLAM_IMPL_HPP

#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class ORBSLAMExtractor : public FeatureBase
{
 public:
  ORBSLAMExtractor() = default;
  virtual bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                      cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat());
};

inline bool ORBSLAMExtractor::detect(const cv::Mat &image,
                                     std::vector<cv::KeyPoint> &kps,
                                     cv::Mat mask, cv::Mat desc) {
  return true;
}

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_ORB_SLAM_IMPL_HPP
