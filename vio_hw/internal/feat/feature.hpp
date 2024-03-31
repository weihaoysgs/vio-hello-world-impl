#ifndef VIO_HELLO_WORLD_FEATURE_HPP
#define VIO_HELLO_WORLD_FEATURE_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

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
    int kps_max_distance_;
    float kps_quality_level_;
  };

  FeatureBase();

  virtual ~FeatureBase() = default;

  // TODO: the [desc] is redundant
  virtual bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                      cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat()) = 0;

  virtual std::vector<cv::Mat> DescribeBRIEF(const cv::Mat &im,
                                             const std::vector<cv::Point2f> &vpts);

  static std::shared_ptr<FeatureBase> Create(const FeatureExtractorOptions &options);

  void setMaxKpsNumber(int num) { max_kps_number_ = num; }

  int max_kps_number_ = 0;
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_desc_extractor_;
};

typedef std::shared_ptr<FeatureBase> FeatureBasePtr;
typedef std::shared_ptr<const FeatureBase> FeatureBaseConstPtr;

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_FEATURE_HPP
