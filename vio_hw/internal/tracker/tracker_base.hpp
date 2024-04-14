#ifndef VIO_HELLO_WORLD_TRACKER_BASE_HPP
#define VIO_HELLO_WORLD_TRACKER_BASE_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace viohw {

class OpticalFlowImplConfig;
class LightGlueImplConfig;

class TrackerBase
{
public:
  enum TrackerType
  {
    OPTICAL_FLOW,
    LIGHT_GLUE,
    SUPER_GLUE,
    BRUTE_FORCE,
    ERR,
  };

  static std::map<std::string, TrackerType> trackerNames;
  static TrackerType StringToTrackerType( const std::string &feat ) {
    auto it = trackerNames.find( feat );
    if ( it != trackerNames.end() ) {
      return it->second;
    } else {
      return ERR;
    }
  }

  struct TrackerOption
  {
    TrackerType tracker_type_;
    std::shared_ptr<OpticalFlowImplConfig> opticalFlowImplConfig;
    std::shared_ptr<LightGlueImplConfig> lightGlueImplConfig;
  };

  virtual ~TrackerBase() = default;

  virtual void trackerAndMatcher( const std::vector<cv::Mat> &, const std::vector<cv::Mat> &,
                                  std::vector<cv::Point2f> &, std::vector<cv::Point2f> &,
                                  std::vector<bool> &,
                                  Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                                  Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 ) = 0;

  static std::shared_ptr<TrackerBase> Create( const TrackerOption &options );
};

inline std::map<std::string, TrackerBase::TrackerType> TrackerBase::trackerNames = {
    { "OPTICAL_FLOW", OPTICAL_FLOW },
    { "LIGHT_GLUE", LIGHT_GLUE },
    { "SUPER_GLUE", SUPER_GLUE },
    { "BRUTE_FORCE", BRUTE_FORCE } };

typedef std::shared_ptr<TrackerBase> TrackerBasePtr;
typedef std::shared_ptr<const TrackerBase> TrackerBaseConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_TRACKER_BASE_HPP
