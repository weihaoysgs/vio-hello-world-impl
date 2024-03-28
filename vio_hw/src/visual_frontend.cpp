#include "vio_hw/internal/visual_frontend.hpp"

namespace viohw {
VisualFrontEnd::VisualFrontEnd(viohw::SettingPtr state, viohw::FramePtr frame,
                               viohw::MapManagerPtr map,
                               viohw::TrackerBasePtr tracker)
    : param_(state),
      current_frame_(frame),
      map_manager_(map),
      tracker_(tracker) {}

}  // namespace viohw
