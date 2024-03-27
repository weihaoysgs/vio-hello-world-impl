#include "vio_hw/internal/viz/visualization_base.hpp"

#include <glog/logging.h>

#include "vio_hw/internal/viz/pangolin_visualization.hpp"
#include "vio_hw/internal/viz/rviz_visualization.hpp"

namespace viohw {
std::shared_ptr<VisualizationBase> VisualizationBase::Create(
    const VisualizationBase::VisualizationOption& options) {
  switch (options.viz_type_) {
    case RVIZ: {
      return std::make_shared<RvizVisualization>();
    }
    case PANGOLIN: {
      return std::make_shared<PangolinVisualization>();
    }
    default: {
      LOG(FATAL) << "Please select a correct ui tool";
    }
  }
}
}  // namespace viohw
