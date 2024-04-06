#ifndef VIO_HELLO_WORLD_ORB_SLAM_IMPL_HPP
#define VIO_HELLO_WORLD_ORB_SLAM_IMPL_HPP

#include "feat/orb_slam/orbextractor.hpp"
#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class ORBSLAMExtractorConfig
{
public:
  int max_kps_ = 300;
  float scale_factor_ = 1.2;
  int level_ = 8;
  int iniThFAST_ = 20;
  int minThFAST_ = 7;
};

typedef std::shared_ptr<ORBSLAMExtractorConfig> ORBSLAMExtractorConfigPtr;
typedef std::shared_ptr<const ORBSLAMExtractorConfig> ORBSLAMExtractorConfigConstPtr;

class ORBSLAMExtractor : public FeatureBase
{
public:
  explicit ORBSLAMExtractor(const ORBSLAMExtractorConfig& option);
  virtual bool detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                       cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat() );
  std::shared_ptr<feat::ORBextractor> orb_extractor_ = nullptr;
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_ORB_SLAM_IMPL_HPP
