#include <glog/logging.h>

#include "vio_hw/internal/feat/feature.hpp"
#include "vio_hw/internal/feat/good_feature_impl.hpp"
#include "vio_hw/internal/feat/orb_cv_impl.hpp"
#include "vio_hw/internal/feat/orb_slam_impl.hpp"

namespace viohw {

FeatureBase::FeatureBase() {
  brief_desc_extractor_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
}

std::shared_ptr<FeatureBase> FeatureBase::Create(const FeatureExtractorOptions& options) {
  switch (options.feature_type_) {
    case FeatureType::ORB_CV: {
      LOG(INFO) << "Create Feature Extractor with [ORB_CV]";
      return std::make_shared<ORBCVExtractor>(options);
    }
    case FeatureType::HARRIS: {
      LOG(INFO) << "Create Feature Extractor with [HARRIS]";
      return std::make_shared<GoodFeature2Tracker>(*options.goodFeature2TrackerConfig);
    }
    case FeatureType::ORB: {
      LOG(INFO) << "Create Feature Extractor with [ORB]";
      return std::make_shared<ORBSLAMExtractor>(*options.orbslamExtractorConfig);
    }
    case FeatureType::SUPER_POINT: {
      // TODO
      LOG(FATAL) << "TODO Feature Extractor with [SUPER_POINT].";
      break;
    }
    default: {
      LOG(FATAL) << "Please select correct feature detector method.";
    }
  }
}

std::vector<cv::Mat> FeatureBase::DescribeBRIEF(const cv::Mat& im,
                                                const std::vector<cv::Point2f>& vpts) {
  if (vpts.empty()) {
    return {};
  }
  assert(brief_desc_extractor_ != nullptr);
  std::vector<cv::KeyPoint> vkps;
  size_t nbkps = vpts.size();
  vkps.reserve(nbkps);
  std::vector<cv::Mat> vdescs;
  vdescs.reserve(nbkps);

  cv::KeyPoint::convert(vpts, vkps);

  cv::Mat descs;
  brief_desc_extractor_->compute(im, vkps, descs);

  if (vkps.empty()) {
    return std::vector<cv::Mat>(nbkps, cv::Mat());
  }
  size_t k = 0;
  for (size_t i = 0; i < nbkps; i++) {
    if (k < vkps.size()) {
      if (vkps[k].pt == vpts[i]) {
        // vdescs.push_back(descs.row(k).clone());
        vdescs.push_back(descs.row(k));
        k++;
      } else {
        vdescs.push_back(cv::Mat());
      }
    } else {
      vdescs.push_back(cv::Mat());
    }
  }
  assert(vdescs.size() == vpts.size());
  return vdescs;
}

}  // namespace viohw