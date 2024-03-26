#include <glog/logging.h>

#include "vio_hw/internal/feat/feature.hpp"
#include "vio_hw/internal/feat/good_feature_impl.hpp"
#include "vio_hw/internal/feat/orb_cv_impl.hpp"
#include "vio_hw/internal/feat/orb_slam_impl.hpp"

namespace viohw {

std::shared_ptr<FeatureBase> FeatureBase::Create(
    const FeatureExtractorOptions& options) {
  switch (options.feature_type_) {
    case FeatureType::ORB_CV: {
      return std::make_shared<ORBCVExtractor>();
    }
    case FeatureType::HARRIS: {
      return std::make_shared<GoodFeature2Tracker>(options);
    }
    case FeatureType::ORB: {
      return std::make_shared<ORBSLAMExtractor>();
    }
    case FeatureType::SUPER_POINT: {
      // TODO
      break;
    }
    default: {
      LOG(FATAL) << "Please select correct feature detector method.";
    }
  }
}

}  // namespace viohw