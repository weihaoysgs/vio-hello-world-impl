#ifndef VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP
#define VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP

#include "feat/harris/harris.h"
#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class GoodFeature2TrackerConfig
{
public:
  int kps_min_distance_;
  float kps_quality_level_;
  int max_kps_num_;
};

typedef std::shared_ptr<GoodFeature2TrackerConfig> GoodFeature2TrackerConfigPtr;
typedef std::shared_ptr<const GoodFeature2TrackerConfig> GoodFeature2TrackerConfigConstPtr;

class GoodFeature2Tracker : public FeatureBase
{
public:
  explicit GoodFeature2Tracker( const GoodFeature2TrackerConfig &options );

  bool detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat mask = cv::Mat(),
               cv::Mat desc = cv::Mat() ) override;

private:
  GoodFeature2TrackerConfig feature_extract_config_;
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP
