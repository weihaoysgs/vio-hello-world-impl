#include "vio_hw/internal/feat/orb_slam_impl.hpp"

namespace viohw {
bool ORBSLAMExtractor::detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &mask,
                               cv::Mat &desc, Eigen::Matrix<double, 259, Eigen::Dynamic> &feat ) {
  orb_extractor_->Detect( image, mask, kps );
  return true;
}

ORBSLAMExtractor::ORBSLAMExtractor( const ORBSLAMExtractorConfig &option ) {
  orb_extractor_.reset( new feat::ORBextractor( option.max_kps_, option.scale_factor_,
                                                option.level_, option.iniThFAST_,
                                                option.minThFAST_ ) );
}

}  // namespace viohw