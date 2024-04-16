#ifndef VIO_HELLO_WORLD_ESTIMATOR_HPP
#define VIO_HELLO_WORLD_ESTIMATOR_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <thread>

#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/optimization.hpp"
#include "vio_hw/internal/setting.hpp"
#include "vio_hw/internal/system_status.hpp"

namespace viohw {
class Estimator
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // constructor
  Estimator() = default;

  Estimator( SettingPtr param, MapManagerPtr map_manager, OptimizationPtr optimization, SystemStatePtr state );

  // backend estimator thread
  void run();

  void ApplyLocalBA();

  bool GetNewKf();

  void AddNewKf( const std::shared_ptr<Frame> &kf );

private:
  SettingPtr param_;
  MapManagerPtr map_manager_;
  OptimizationPtr optimization_;
  SystemStatePtr system_state_;

  bool is_newkf_available_ = false;
  bool is_loose_ba_on_ = false;

  std::shared_ptr<Frame> newkf_;
  std::queue<std::shared_ptr<Frame>> queen_kfs_;
  std::mutex qkf_mutex_;
};

typedef std::shared_ptr<Estimator> EstimatorPtr;
typedef std::shared_ptr<const Estimator> EstimatorConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_ESTIMATOR_HPP
