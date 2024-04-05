#include "vio_hw/internal/optimization.hpp"

namespace viohw {
viohw::Optimization::Optimization( viohw::SettingPtr param, viohw::MapManagerPtr map_manager )
    : map_manager_( map_manager ), param_( param ) {}

bool Optimization::LocalPoseGraph( Frame& new_frame, int kf_loop_id, const Sophus::SE3d& new_Twc ) {
  tceres::Problem problem;

  std::map<int, backend::PoseParametersBlock> map_id_poses_param_block;
  std::map<int, std::shared_ptr<Frame>> map_kfs;

  Sophus::SE3d init_Twc = new_frame.GetTwc();
  auto loop_kf = map_manager_->GetKeyframe( kf_loop_id );
  CHECK( loop_kf != nullptr ) << "Loop KF empty";

  // add loop kf pose to problem and set constant
  Sophus::SE3d T_loop_wc = loop_kf->GetTwc();
  map_id_poses_param_block.emplace( kf_loop_id,
                                    backend::PoseParametersBlock( kf_loop_id, T_loop_wc ) );
  auto* local_pose_parameterization = new backend::SE3LeftParameterization;
  problem.AddParameterBlock( map_id_poses_param_block.at( kf_loop_id ).values(),
                             backend::PoseParametersBlock::ndim_, local_pose_parameterization );
  problem.SetParameterBlockConstant( map_id_poses_param_block.at( kf_loop_id ).values() );

  Sophus::SE3d T_ci_w = loop_kf->GetTcw();
  int ci_kf_id = loop_kf->kfid_;
  Sophus::SE3d T_loopkf_newkf = T_ci_w * new_Twc;

  // correct loopKF -> currKF pose
  for ( int kfid = kf_loop_id + 1; kfid <= new_frame.kfid_; kfid++ ) {
    auto kf = map_manager_->GetKeyframe( kfid );

    if ( kf == nullptr ) {
      if ( kfid == new_frame.kfid_ ) {
        return false;
      } else {
        continue;
      }
    }

    map_kfs.emplace( kfid, kf );
    Sophus::SE3d T_w_cj = kf->GetTwc();

    map_id_poses_param_block.emplace( kfid, backend::PoseParametersBlock( kfid, T_w_cj ) );
    auto* local_parameterization = new backend::SE3LeftParameterization;
    problem.AddParameterBlock( map_id_poses_param_block.at( kfid ).values(),
                               backend::PoseParametersBlock::ndim_, local_parameterization );
    Sophus::SE3d T_ij = T_ci_w * T_w_cj;
    tceres::CostFunction* cost_function = new backend::LeftSE3RelativePoseError( T_ij );
    problem.AddResidualBlock( cost_function, nullptr,
                              map_id_poses_param_block.at( ci_kf_id ).values(),
                              map_id_poses_param_block.at( kfid ).values() );
    T_ci_w = T_w_cj.inverse();
    ci_kf_id = kfid;
  }

  tceres::CostFunction* cost_fun = new backend::LeftSE3RelativePoseError( T_loopkf_newkf );
  problem.AddResidualBlock( cost_fun, nullptr, map_id_poses_param_block.at( kf_loop_id ).values(),
                            map_id_poses_param_block.at( new_frame.kfid_ ).values() );

  backend::examples::OutputPoses("./initial_poses.txt", map_id_poses_param_block);

  tceres::Solver::Options options;
  options.linear_solver_type = tceres::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10;
  options.function_tolerance = 1.e-4;

  tceres::Solver::Summary summary;
  tceres::Solve( options, &problem, &summary );
  LOG( INFO ) << summary.FullReport();

  backend::examples::OutputPoses("./optimization_poses.txt", map_id_poses_param_block);

  return true;
}

}  // namespace viohw
