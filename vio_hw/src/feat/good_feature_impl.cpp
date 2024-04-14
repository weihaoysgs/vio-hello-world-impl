#include "vio_hw/internal/feat/good_feature_impl.hpp"

namespace viohw {
GoodFeature2Tracker::GoodFeature2Tracker( const GoodFeature2TrackerConfig &options ) {
  feature_extract_config_ = options;
  tobe_extractor_kps_num_ = options.max_kps_num_;
}

bool GoodFeature2Tracker::detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                                  cv::Mat &mask, cv::Mat &desc,
                                  Eigen::Matrix<double, 259, Eigen::Dynamic> &feat ) {
  std::vector<cv::Point2f> points;
  feat::goodFeaturesToTrack( image, points, tobe_extractor_kps_num_,
                             feature_extract_config_.kps_quality_level_,
                             feature_extract_config_.kps_min_distance_, mask );
  cv::KeyPoint::convert( points, kps );
  return true;
}
}  // namespace viohw
