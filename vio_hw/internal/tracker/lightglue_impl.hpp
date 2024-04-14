#ifndef VIO_HELLO_WORLD_LIGHTGLUE_IMPL_HPP
#define VIO_HELLO_WORLD_LIGHTGLUE_IMPL_HPP
#include "vio_hw/internal/tracker/tracker_base.hpp"

#ifdef ENABLE_DFM
#include "dfm/internal/lightglue.hpp"
#include "dfm/internal/parameter.hpp"
#endif

namespace viohw {

class LightGlueImplConfig
{
public:
  std::string config_file_path_;
  int max_kps_num_;
};

class LightGlueImpl : public TrackerBase
{
public:
  explicit LightGlueImpl( const LightGlueImplConfig &config );

  void trackerAndMatcher( const std::vector<cv::Mat> &, const std::vector<cv::Mat> &,
                          std::vector<cv::Point2f> &, std::vector<cv::Point2f> &,
                          std::vector<bool> &,
                          Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                          Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 ) override;

private:
#ifdef ENABLE_DFM
  dfm::SuperPointConfig sp_config_;
  dfm::LightGlueConfig lg_config_;
  std::shared_ptr<dfm::LightGlue> lightglue_infer_;
  std::shared_ptr<dfm::SuperPoint> superpoint_infer_;
  std::mutex sp_memory_lock_;
#endif
};

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_LIGHTGLUE_IMPL_HPP
