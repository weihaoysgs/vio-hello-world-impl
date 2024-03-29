#ifndef VIO_HELLO_WORLD_WORLD_MANAGER_HPP
#define VIO_HELLO_WORLD_WORLD_MANAGER_HPP

#include <chrono>
#include <thread>

#include "vio_hw/internal/feat/feature.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"
#include "vio_hw/internal/visual_frontend.hpp"
#include "vio_hw/internal/viz/visualization_base.hpp"

namespace viohw {
class WorldManager
{
 public:
  explicit WorldManager(std::shared_ptr<Setting> &setting);
  void run();
  const std::shared_ptr<Setting> getParams() const { return params_; }
  void addNewStereoImages(double time, cv::Mat &im0, cv::Mat &im1);
  bool getNewImage(cv::Mat &iml, cv::Mat &imr, double &time);
  void setupCalibration();

 private:
  std::queue<cv::Mat> img_left_queen_, img_right_queen_;
  std::queue<double> img_time_queen_;

  std::mutex img_mutex_;

  bool is_new_img_available_ = false;
  int frame_id_ = -1;

  FeatureBasePtr feature_extractor_;
  TrackerBasePtr tracker_;
  FramePtr current_frame_;
  MapManagerPtr map_manager_;
  VisualFrontEndConstPtr visual_frontend_;
  std::shared_ptr<VisualizationBase> viz_;
  std::shared_ptr<CameraCalibration> calib_model_left_, calib_model_right_;
  const std::shared_ptr<Setting> params_;
};
}  // namespace viohw
#endif  // VIO_HELLO_WORLD_WORLD_MANAGER_HPP
