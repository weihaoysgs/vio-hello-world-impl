#ifndef VIO_HELLO_WORLD_ORBCV_SLAM_IMPL_HPP
#define VIO_HELLO_WORLD_ORBCV_SLAM_IMPL_HPP

#include "feat/orb/orb_feature.h"
#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class ORBCVExtractor : public FeatureBase
{
 public:
  explicit ORBCVExtractor(const FeatureBase::FeatureExtractorOptions &options);
  bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &mask,
               cv::Mat &desc, Eigen::Matrix<double, 259, Eigen::Dynamic> &feat) override;

 private:
  cv::Ptr<feat::ORB> orb_;
};

}  // namespace viohw
#endif