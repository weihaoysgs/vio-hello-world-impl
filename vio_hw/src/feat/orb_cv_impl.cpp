#include "vio_hw/internal/feat/orb_cv_impl.hpp"
namespace viohw {

bool ORBCVExtractor::detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &mask,
                             cv::Mat &desc, Eigen::Matrix<double, 259, Eigen::Dynamic> &feat ) {
  orb_->detectAndCompute( image, mask, kps, desc );
  return true;
}

ORBCVExtractor::ORBCVExtractor( const FeatureBase::FeatureExtractorOptions &options ) {
  // orb_ = feat::ORB::create(options.max_kps_num_);
}

}  // namespace viohw