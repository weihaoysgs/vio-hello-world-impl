#ifndef VIO_HELLO_WORLD_SUPERPOINT_IMPL_HPP
#define VIO_HELLO_WORLD_SUPERPOINT_IMPL_HPP

#include "vio_hw/internal/feat/feature.hpp"

namespace viohw {
class SuperPointExtractorConfig
{
public:
  std::string config_file_path_;
  int max_kps_;
};
}  // namespace viohw

#ifdef ENABLE_DFM
#include "dfm/internal/parameter.hpp"
#include "dfm/internal/superpoint.hpp"
#include "opencv2/core/eigen.hpp"

namespace viohw {

class SuperPointImpl : public FeatureBase
{
public:
  SuperPointImpl() = default;
  explicit SuperPointImpl( const SuperPointExtractorConfig &config );
  ~SuperPointImpl() override = default;
  bool detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &mask, cv::Mat &desc,
               Eigen::Matrix<double, 259, Eigen::Dynamic> &feat ) override;

private:
  dfm::SuperPointConfig sp_config_;
  dfm::LightGlueConfig lg_config_;
  std::shared_ptr<dfm::SuperPoint> superpoint_infer_;
  int max_kps_ = 0;
};

typedef std::shared_ptr<SuperPointImpl> SuperPointImplPtr;
typedef std::shared_ptr<const SuperPointImpl> SuperPointImplConstPtr;
}  // namespace viohw

#endif
#endif  // VIO_HELLO_WORLD_SUPERPOINT_IMPL_HPP
