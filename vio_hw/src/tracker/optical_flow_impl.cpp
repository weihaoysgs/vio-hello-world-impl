#include "vio_hw/internal/tracker/optical_flow_impl.hpp"

#include <glog/logging.h>

namespace viohw {

void OpticalFlowImpl::trackerAndMatcher(cv::InputArray prevImg,
                                        cv::InputArray nextImg,
                                        cv::InputArray prevPts,
                                        cv::InputOutputArray nextPts,
                                        cv::OutputArray status) {
  std::cout << "OpticalFlowImpl::trackerAndMatcher\n";
}

void OpticalFlowImpl::trackerAndMatcher(
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features1,
    std::vector<cv::DMatch>& matches, bool outlier_rejection) {
  LOG(WARNING) << "Nothing todo in OpticalFlowImpl";
}
}  // namespace viohw
