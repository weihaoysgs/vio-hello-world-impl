#ifndef VIO_HELLO_WORLD_MAP_MANAGER_HPP
#define VIO_HELLO_WORLD_MAP_MANAGER_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "vio_hw/internal/feat/feature.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"

namespace viohw {

class MapManager
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MapManager(SettingPtr state, FramePtr pframe, FeatureBasePtr pfeatextract,
             TrackerBasePtr ptracker);
  ~MapManager() = default;

 private:
  SettingPtr param_;
  FramePtr current_frame_;
  FeatureBasePtr feature_extractor_;
  TrackerBasePtr tracker_;
};

typedef std::shared_ptr<MapManager> MapManagerPtr;
typedef std::shared_ptr<const MapManager> MapManagerConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_MAP_MANAGER_HPP
