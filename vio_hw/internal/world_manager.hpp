#ifndef VIO_HELLO_WORLD_WORLD_MANAGER_HPP
#define VIO_HELLO_WORLD_WORLD_MANAGER_HPP

#include <chrono>
#include <fstream>
#include <thread>

#include "backend/sensor_fusion/imu/imu_database.h"
#include "backend/sensor_fusion/imu/imu_types.h"
#include "vio_hw/internal/estimator.hpp"
#include "vio_hw/internal/feat/feature.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/mapping.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/tracker/tracker_base.hpp"
#include "vio_hw/internal/visual_frontend.hpp"
#include "vio_hw/internal/viz/visualization_base.hpp"

namespace viohw {
class WorldManager
{
public:
  explicit WorldManager( std::shared_ptr<Setting> &setting );
  void run();
  const std::shared_ptr<Setting> getParams() const { return params_; }
  void addNewStereoImages( double time, cv::Mat &im0, cv::Mat &im1 );
  bool getNewImage( cv::Mat &iml, cv::Mat &imr, double &time );
  void setupCalibration();
  bool VisualizationImage();
  void Visualization();
  void VisualizationKFTraj();
  bool GenerateFeatureExtractorBase();
  bool GenerateFeatureTrackerMatcherBase();
  void SaveKFTrajectoryTUM( const std::string path );
  void InsertIMUMeasure( backend::IMU::Point &data );
  void PreIntegrateIMU( std::vector<backend::IMU::Point> &imu_data, double last_image_time,
                        double curr_image_time );

private:
  std::queue<cv::Mat> img_left_queen_, img_right_queen_;
  std::queue<double> img_time_queen_;

  std::mutex img_mutex_;

  bool is_new_img_available_ = false;
  bool kf_viz_is_on_ = false;
  int frame_id_ = -1;

  FeatureBasePtr feature_extractor_;
  TrackerBasePtr tracker_, tracker_for_mapping_;
  FramePtr current_frame_;
  MapManagerPtr map_manager_;
  VisualFrontEndPtr visual_frontend_;
  MappingPtr mapping_;
  LoopCloserPtr loop_closer_;
  OptimizationPtr optimization_;
  EstimatorPtr estimator_;

  std::shared_ptr<VisualizationBase> viz_;
  std::shared_ptr<CameraCalibration> calib_model_left_, calib_model_right_;
  const std::shared_ptr<Setting> params_;

  // imu database
  backend::IMU::IMUDataBasePtr imu_database_;
  bool is_init_imu_pose_ = false;
};

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimeStamp;

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_WORLD_MANAGER_HPP
