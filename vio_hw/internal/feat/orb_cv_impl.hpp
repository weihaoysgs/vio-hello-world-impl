//
// Created by weihao on 24-3-26.
//
#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {

class ORBCVExtractor : public FeatureBase
{
 public:
  ORBCVExtractor() = default;
  virtual bool detect(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                      cv::Mat mask = cv::Mat(), cv::Mat desc = cv::Mat());
};

bool ORBCVExtractor::detect(const cv::Mat &image,
                            std::vector<cv::KeyPoint> &kps, cv::Mat mask,
                            cv::Mat) {
  return true;
}

}  // namespace viohw
