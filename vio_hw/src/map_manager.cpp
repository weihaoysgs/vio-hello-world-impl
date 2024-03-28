#include "vio_hw/internal/map_manager.hpp"

namespace viohw {
MapManager::MapManager(SettingPtr state, FramePtr frame,
                       FeatureBasePtr feat_extract, TrackerBasePtr tracker)
    : param_(state),
      current_frame_(frame),
      feature_extractor_(feat_extract),
      tracker_(tracker) {}

}  // namespace viohw
