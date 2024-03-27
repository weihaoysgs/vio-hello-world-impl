#include "vio_hw/internal/feat/good_feature_impl.hpp"

namespace viohw {
GoodFeature2Tracker::GoodFeature2Tracker(
    const FeatureBase::FeatureExtractorOptions &options) {
  max_kps_num_ = options.max_kps_num_;
  kps_min_distance_ = options.kps_min_distance_;
  kps_quality_level_ = options.kps_quality_level_;
}

bool GoodFeature2Tracker::detect(const cv::Mat &image,
                                 std::vector<cv::KeyPoint> &kps, cv::Mat mask,
                                 cv::Mat desc) {
  std::vector<cv::Point2f> points;
  feat::goodFeaturesToTrack(image, points, max_kps_num_, kps_quality_level_,
                            kps_min_distance_, cv::Mat());
  cv::KeyPoint::convert(points, kps);
  return true;
}
}  // namespace viohw
