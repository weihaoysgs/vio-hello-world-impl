#ifndef VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP
#define VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP

#include "feat/harris/harris.h"
#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class GoodFeature2Tracker : public FeatureBase
{
 public:
  explicit GoodFeature2Tracker(const FeatureBase::FeatureExtractorOptions &options);

  bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
              cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat()) override;

 private:
  int max_kps_num_;
  int kps_min_distance_;
  float kps_quality_level_;
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP
