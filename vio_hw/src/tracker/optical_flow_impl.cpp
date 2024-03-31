#include "vio_hw/internal/tracker/optical_flow_impl.hpp"

#include <glog/logging.h>

#include "feat/optical_flow/lkpyramid.h"

namespace viohw {

void OpticalFlowImpl::trackerAndMatcher(const std::vector<cv::Mat> &vprevpyr,
                                        const std::vector<cv::Mat> &vcurpyr, int nwinsize,
                                        int nbpyrlvl, float ferr, float fmax_fbklt_dist,
                                        std::vector<cv::Point2f> &vkps,
                                        std::vector<cv::Point2f> &vpriorkps,
                                        std::vector<bool> &vkpstatus) {
  // TODO
  cv::TermCriteria klt_convg_crit_(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.01f);

  assert(vprevpyr.size() == vcurpyr.size());

  if (vkps.empty()) {
    // std::cout << "\n \t >>> No kps were provided to kltTracking()!\n";
    return;
  }

  cv::Size klt_win_size(nwinsize, nwinsize);

  if ((int)vprevpyr.size() < 2 * (nbpyrlvl + 1)) {
    nbpyrlvl = vprevpyr.size() / 2 - 1;
  }

  // Objects for OpenCV KLT
  size_t nbkps = vkps.size();
  vkpstatus.reserve(nbkps);

  std::vector<uchar> vstatus;
  std::vector<float> verr;
  std::vector<int> vkpsidx;
  vstatus.reserve(nbkps);
  verr.reserve(nbkps);
  vkpsidx.reserve(nbkps);

  // Tracking Forward
  feat::calcOpticalFlowPyrLK(vprevpyr, vcurpyr, vkps, vpriorkps, vstatus, verr, klt_win_size,
                             nbpyrlvl, klt_convg_crit_,
                             (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

  std::vector<cv::Point2f> vnewkps;
  std::vector<cv::Point2f> vbackkps;
  vnewkps.reserve(nbkps);
  vbackkps.reserve(nbkps);

  size_t nbgood = 0;

  // Init outliers vector & update tracked kps
  for (size_t i = 0; i < nbkps; i++) {
    if (!vstatus.at(i)) {
      vkpstatus.push_back(false);
      continue;
    }

    if (verr.at(i) > ferr) {
      vkpstatus.push_back(false);
      continue;
    }

    if (!InBorder(vpriorkps.at(i), vcurpyr.at(0))) {
      vkpstatus.push_back(false);
      continue;
    }

    vnewkps.push_back(vpriorkps.at(i));
    vbackkps.push_back(vkps.at(i));
    vkpstatus.push_back(true);
    vkpsidx.push_back(i);
    nbgood++;
  }

  if (vnewkps.empty()) {
    return;
  }

  vstatus.clear();
  verr.clear();

  // std::cout << "\n \t >>> Forward kltTracking : #" << nbgood << " out of #" << nbkps << " \n";

  // Tracking Backward
  feat::calcOpticalFlowPyrLK(vcurpyr, vprevpyr, vnewkps, vbackkps, vstatus, verr, klt_win_size, 0,
                             klt_convg_crit_,
                             (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS));

  nbgood = 0;
  for (int i = 0, iend = vnewkps.size(); i < iend; i++) {
    int idx = vkpsidx.at(i);

    if (!vstatus.at(i)) {
      vkpstatus.at(idx) = false;
      continue;
    }

    if (cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist) {
      vkpstatus.at(idx) = false;
      continue;
    }

    nbgood++;
  }
}

void OpticalFlowImpl::trackerAndMatcher(Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                                        Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                                        std::vector<cv::DMatch> &matches, bool outlier_rejection) {
  LOG(WARNING) << "Nothing todo in OpticalFlowImpl";
}

bool OpticalFlowImpl::InBorder(const cv::Point2f &pt, const cv::Mat &im) {
  const float BORDER_SIZE = 1.;

  return BORDER_SIZE <= pt.x && pt.x < im.cols - BORDER_SIZE && BORDER_SIZE <= pt.y &&
         pt.y < im.rows - BORDER_SIZE;
}
}  // namespace viohw
