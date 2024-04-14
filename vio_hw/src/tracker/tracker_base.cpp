#include "vio_hw/internal/tracker/tracker_base.hpp"

#include <glog/logging.h>

#include "vio_hw/internal/tracker/lightglue_impl.hpp"
#include "vio_hw/internal/tracker/optical_flow_impl.hpp"

namespace viohw {

std::shared_ptr<TrackerBase> TrackerBase::Create(
    const TrackerBase::TrackerOption& options) {
  switch (options.tracker_type_) {
    case OPTICAL_FLOW: {
      LOG(INFO) << "Create Tracker with [OPTICAL_FLOW]";
      return std::make_shared<OpticalFlowImpl>(*options.opticalFlowImplConfig);
    }
    case LIGHT_GLUE: {
    #ifdef ENABLE_DFM
      LOG(INFO) << "Create Tracker with [LIGHT_GLUE]";
      return std::make_shared<LightGlueImpl>(*options.lightGlueImplConfig);
    #else
      LOG(FATAL) << "DFM-Matcher Not Enable, Create Tracker with [LIGHT_GLUE] Failed";
    #endif
    }
    case SUPER_GLUE: {
      LOG(FATAL) << "TOBE support SUPER_GLUE";
    }
    case BRUTE_FORCE: {
      LOG(FATAL) << "TOBE support BRUTE_FORCE";
    }
    default: {
      LOG(FATAL) << "Please select correct tracker/matcher method";
    }
  }
}
}  // namespace viohw