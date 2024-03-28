#ifndef VIO_HELLO_WORLD_FEATURE_HPP
#define VIO_HELLO_WORLD_FEATURE_HPP

#include <opencv2/opencv.hpp>

namespace viohw {

class FeatureBase
{
 public:
  enum FeatureType
  {
    ORB_CV,
    HARRIS,
    ORB,
    SUPER_POINT
  };

  struct FeatureExtractorOptions
  {
    FeatureType feature_type_;
    int max_kps_num_;
    int kps_min_distance_;
    float kps_quality_level_;
  };

  virtual ~FeatureBase() = default;

  // TODO: the [desc] is redundant
  virtual bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                      cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat()) = 0;

  static std::shared_ptr<FeatureBase> Create(
      const FeatureExtractorOptions &options);

  static FeatureBase::FeatureExtractorOptions getDefaultOptions() {
    FeatureBase::FeatureExtractorOptions feature_extractor_options{};
    feature_extractor_options.kps_quality_level_ = 0.03;
    feature_extractor_options.feature_type_ = FeatureBase::HARRIS;
    feature_extractor_options.kps_min_distance_ = 20;
    feature_extractor_options.max_kps_num_ = 200;
    return feature_extractor_options;
  }
};

typedef std::shared_ptr<FeatureBase> FeatureBasePtr;
typedef std::shared_ptr<const FeatureBase> FeatureBaseConstPtr;

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_FEATURE_HPP
