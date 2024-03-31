#include "vio_hw/internal/tracker/lightglue_impl.hpp"

#include <glog/logging.h>

namespace viohw {

void LightGlueImpl::trackerAndMatcher(
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
    std::vector<cv::DMatch>& matches, bool outlier_rejection) {
  std::cout << "LightGlueImpl::trackerAndMatcher\n";
}

void LightGlueImpl::trackerAndMatcher(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr,
                                      int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist, std::vector<cv::Point2f> &vkps,
                                      std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) {
  LOG(WARNING) << "Nothing todo in LightGlueImpl\n";
}
}  // namespace viohw
