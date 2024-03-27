#ifndef VIO_HELLO_WORLD_TRACKER_BASE_HPP
#define VIO_HELLO_WORLD_TRACKER_BASE_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace viohw {

class TrackerBase
{
 public:
  enum TrackerType
  {
    OPTICAL_FLOW,
    LIGHT_GLUE,
    SUPER_GLUE,
    BRUTE_FORCE
  };

  struct TrackerOption
  {
    TrackerType tracker_type_;
  };

  virtual ~TrackerBase() = default;

  virtual void trackerAndMatcher(cv::InputArray prevImg, cv::InputArray nextImg,
                                 cv::InputArray prevPts,
                                 cv::InputOutputArray nextPts,
                                 cv::OutputArray status){};

  virtual void trackerAndMatcher(
      Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
      Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
      std::vector<cv::DMatch> &matches, bool outlier_rejection){};

  static std::shared_ptr<TrackerBase> Create(const TrackerOption &options);
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_TRACKER_BASE_HPP
