#ifndef VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP
#define VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP

#include "vio_hw/internal/tracker/tracker_base.hpp"

namespace viohw {

class OpticalFlowImpl : public TrackerBase
{
 public:
  OpticalFlowImpl() = default;
  void trackerAndMatcher(const std::vector<cv::Mat> &, const std::vector<cv::Mat> &, int, int,
                         float, float, std::vector<cv::Point2f> &, std::vector<cv::Point2f> &,
                         std::vector<bool> &) override;
  void trackerAndMatcher(Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                         Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                         std::vector<cv::DMatch> &matches, bool outlier_rejection) override;
  static bool InBorder(const cv::Point2f &pt, const cv::Mat &im) ;
};

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP
