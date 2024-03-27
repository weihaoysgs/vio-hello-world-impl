#include "vio_hw/internal/tracker/lightglue_impl.hpp"

#include <glog/logging.h>

namespace viohw {

void LightGlueImpl::trackerAndMatcher(
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
    std::vector<cv::DMatch>& matches, bool outlier_rejection) {
  std::cout << "LightGlueImpl::trackerAndMatcher\n";
}

void LightGlueImpl::trackerAndMatcher(cv::InputArray prevImg,
                                      cv::InputArray nextImg,
                                      cv::InputArray prevPts,
                                      cv::InputOutputArray nextPts,
                                      cv::OutputArray status) {
  LOG(WARNING) << "Nothing todo in LightGlueImpl\n";
}
}  // namespace viohw
