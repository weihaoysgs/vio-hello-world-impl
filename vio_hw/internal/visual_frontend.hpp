#ifndef VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP
#define VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP

#include "vio_hw/internal/constant_motion_model.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"

namespace viohw {
class VisualFrontEnd
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VisualFrontEnd() = default;
  ~VisualFrontEnd() = default;
  VisualFrontEnd(SettingPtr setting, FramePtr frame, MapManagerPtr manager,
                 TrackerBasePtr tracker);

 private:
  MapManagerPtr map_manager_;
  FramePtr current_frame_;
  SettingPtr param_;
  TrackerBasePtr tracker_;
};

typedef std::shared_ptr<VisualFrontEnd> VisualFrontEndPtr;
typedef std::shared_ptr<const VisualFrontEnd> VisualFrontEndConstPtr;

}  // namespace viohw


#endif  // VIO_HELLO_WORLD_VISUAL_FRONTEND_HPP
