#ifndef VIO_HELLO_WORLD_OPTIMIZATION_HPP
#define VIO_HELLO_WORLD_OPTIMIZATION_HPP

#include <glog/logging.h>

#include <Eigen/Core>

#include "backend/cost_function/se3_left_pgo_factor.hpp"
#include "backend/example/common/read_g2o_format.hpp"
#include "backend/parameter_block/se3_parameter_block.hpp"
#include "backend/parameter_block/point_parameter_block.hpp"
#include "backend/parameter_block/camera_intrinsic_parameter_block.hpp"
#include "backend/parameterization/se3left_parametrization.hpp"
#include "backend/cost_function/reprojection_se3_k_xyz_factor.hpp"
#include "tceres/problem.h"
#include "tceres/solver.h"
#include "tceres/stringprintf.h"
#include "tceres/loss_function.h"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"

namespace viohw {

class Optimization
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Optimization() = delete;

  ~Optimization() = default;

  Optimization( SettingPtr param, MapManagerPtr map_manager );

  bool LocalPoseGraph( Frame &new_frame, int kf_loop_id, const Sophus::SE3d &new_Twc );
  bool localPoseGraph( Frame &new_frame, int kf_loop_id, const Sophus::SE3d &new_Twc );

  void LocalBA( Frame &newframe, bool buse_robust_cost );

private:
  SettingPtr param_;
  MapManagerPtr map_manager_;
};

typedef std::shared_ptr<Optimization> OptimizationPtr;
typedef std::shared_ptr<const Optimization> OptimizationConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_OPTIMIZATION_HPP
