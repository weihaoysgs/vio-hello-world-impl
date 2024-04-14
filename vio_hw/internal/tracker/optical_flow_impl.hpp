#ifndef VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP
#define VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP

#include "vio_hw/internal/tracker/tracker_base.hpp"

namespace viohw {

class OpticalFlowImplConfig
{
public:
  OpticalFlowImplConfig( int _nwinsize, int _nbpyrlvl, float _ferr, float _fmax_fbklt_dist )
      : nwinsize( _nwinsize ),
        nbpyrlvl( _nbpyrlvl ),
        ferr( _ferr ),
        fmax_fbklt_dist( _fmax_fbklt_dist ) {}
  int nwinsize;
  int nbpyrlvl;
  float ferr;
  float fmax_fbklt_dist;
};

class OpticalFlowImpl : public TrackerBase
{
public:
  explicit OpticalFlowImpl( const OpticalFlowImplConfig &config )
      : optical_flow_config_( config ){};

  void trackerAndMatcher( const std::vector<cv::Mat> &, const std::vector<cv::Mat> &,
                          std::vector<cv::Point2f> &, std::vector<cv::Point2f> &,
                          std::vector<bool> &,
                          Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                          Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 ) override;

  static bool InBorder( const cv::Point2f &pt, const cv::Mat &im );

private:
  OpticalFlowImplConfig optical_flow_config_;
};

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_OPTICAL_FLOW_IMPL_HPP
