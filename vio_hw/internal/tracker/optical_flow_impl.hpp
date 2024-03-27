#ifndef VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP
#define VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP

#include "vio_hw/internal/tracker/tracker_base.hpp"

namespace viohw {

class OpticalFlowImpl : public TrackerBase
{
 public:
  OpticalFlowImpl() = default;
  void trackerAndMatcher(cv::InputArray prevImg, cv::InputArray nextImg,
                         cv::InputArray prevPts, cv::InputOutputArray nextPts,
                         cv::OutputArray status) override;
  void trackerAndMatcher(
      Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
      Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
      std::vector<cv::DMatch> &matches, bool outlier_rejection) override;
};

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP
