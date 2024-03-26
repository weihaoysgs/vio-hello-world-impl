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
  virtual bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                      cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat()) = 0;
  static std::shared_ptr<FeatureBase> Create(
      const FeatureExtractorOptions &options);
};
}  // namespace viohw
#endif  // VIO_HELLO_WORLD_FEATURE_HPP
