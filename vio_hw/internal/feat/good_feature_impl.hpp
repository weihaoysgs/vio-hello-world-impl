#ifndef VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP
#define VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP

#include "feat/harris/harris.h"
#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class GoodFeature2Tracker : public FeatureBase
{
 public:
  GoodFeature2Tracker(const FeatureBase::FeatureExtractorOptions &options);

  bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
              cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat()) override;

  static FeatureBase::FeatureExtractorOptions getDefaultOptions();

 private:
  int max_kps_num_;
  int kps_min_distance_;
  float kps_quality_level_;
};
inline bool GoodFeature2Tracker::detect(const cv::Mat &image,
                                        std::vector<cv::KeyPoint> &kps,
                                        cv::Mat mask, cv::Mat desc) {
  std::vector<cv::Point2f> points;
  feat::goodFeaturesToTrack(image, points, max_kps_num_, kps_quality_level_,
                            kps_min_distance_, cv::Mat());
  cv::KeyPoint::convert(points, kps);
  return true;
}

inline FeatureBase::FeatureExtractorOptions
GoodFeature2Tracker::getDefaultOptions() {
  FeatureBase::FeatureExtractorOptions feature_extractor_options{};
  feature_extractor_options.kps_quality_level_ = 0.03;
  feature_extractor_options.feature_type_ = FeatureBase::HARRIS;
  feature_extractor_options.kps_min_distance_ = 20;
  feature_extractor_options.max_kps_num_ = 200;
  return feature_extractor_options;
}

inline GoodFeature2Tracker::GoodFeature2Tracker(
    const FeatureBase::FeatureExtractorOptions &options) {
  max_kps_num_ = options.max_kps_num_;
  kps_min_distance_ = options.kps_min_distance_;
  kps_quality_level_ = options.kps_quality_level_;
}

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_GOOD_FEATURE_IMPL_HPP
