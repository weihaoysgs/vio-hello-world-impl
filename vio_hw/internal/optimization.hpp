#ifndef VIO_HELLO_WORLD_OPTIMIZATION_HPP
#define VIO_HELLO_WORLD_OPTIMIZATION_HPP

#include <glog/logging.h>

#include <Eigen/Core>

#include "backend/cost_function/se3_left_pgo_factor.hpp"
#include "backend/example/common/read_g2o_format.hpp"
#include "backend/parameter_block/se3_parameter_block.hpp"
#include "backend/parameterization/se3left_parametrization.hpp"
#include "tceres/problem.h"
#include "tceres/solver.h"
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

private:
  SettingPtr param_;
  MapManagerPtr map_manager_;
};

typedef std::shared_ptr<Optimization> OptimizationPtr;
typedef std::shared_ptr<const Optimization> OptimizationConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_OPTIMIZATION_HPP
