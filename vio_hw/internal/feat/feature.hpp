#ifndef VIO_HELLO_WORLD_FEATURE_HPP
#define VIO_HELLO_WORLD_FEATURE_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace viohw {

class ORBSLAMExtractorConfig;
class GoodFeature2TrackerConfig;
class SuperPointExtractorConfig;

class FeatureBase
{
public:
  enum FeatureType
  {
    ORB_CV,
    HARRIS,
    ORB,
    SUPER_POINT,
    ERR
  };

  static std::map<std::string, FeatureType> colorNames;

  static FeatureType StringToFeatureType( const std::string &feat ) {
    auto it = colorNames.find( feat );
    if ( it != colorNames.end() ) {
      return it->second;
    } else {
      return ERR;
    }
  }

  struct FeatureExtractorOptions
  {
    FeatureType feature_type_;
    std::shared_ptr<ORBSLAMExtractorConfig> orbslamExtractorConfig;
    std::shared_ptr<GoodFeature2TrackerConfig> goodFeature2TrackerConfig;
    std::shared_ptr<SuperPointExtractorConfig> superPointExtractorConfig;
  };

  FeatureBase();

  virtual ~FeatureBase() = default;

  virtual bool detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &mask,
                       cv::Mat &desc, Eigen::Matrix<double, 259, Eigen::Dynamic> &feat ) = 0;

  virtual std::vector<cv::Mat> DescribeBRIEF( const cv::Mat &im,
                                              const std::vector<cv::Point2f> &vpts );

  static std::shared_ptr<FeatureBase> Create( const FeatureExtractorOptions &options );

  void setTobeExtractKpsNumber( int num ) { tobe_extractor_kps_num_ = num; }

  int tobe_extractor_kps_num_ = 0;
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_desc_extractor_;
};
inline std::map<std::string, FeatureBase::FeatureType> FeatureBase::colorNames = {
    { "ORB_CV", ORB_CV }, { "HARRIS", HARRIS }, { "BLUE", ORB }, { "SUPER_POINT", SUPER_POINT } };

typedef std::shared_ptr<FeatureBase> FeatureBasePtr;
typedef std::shared_ptr<const FeatureBase> FeatureBaseConstPtr;

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_FEATURE_HPP
