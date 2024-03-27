#ifndef VIO_HELLO_WORLD_ORBCV_SLAM_IMPL_HPP
#define VIO_HELLO_WORLD_ORBCV_SLAM_IMPL_HPP

#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class ORBCVExtractor : public FeatureBase
{
 public:
  ORBCVExtractor() = default;
  virtual bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                      cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat());
};

inline bool ORBCVExtractor::detect(const cv::Mat &image,
                                   std::vector<cv::KeyPoint> &kps, cv::Mat mask,
                                   cv::Mat) {
  return true;
}

}  // namespace viohw
#endif