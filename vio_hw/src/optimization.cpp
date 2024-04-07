#include "vio_hw/internal/optimization.hpp"

namespace viohw {
viohw::Optimization::Optimization( viohw::SettingPtr param, viohw::MapManagerPtr map_manager )
    : map_manager_( map_manager ), param_( param ) {}

bool Optimization::LocalPoseGraph( Frame& new_frame, int kf_loop_id, const Sophus::SE3d& new_Twc ) {
  tceres::Problem problem;

  std::map<int, backend::PoseParametersBlock> map_id_poses_param_block;
  std::map<int, std::shared_ptr<Frame>> map_kfs;

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

  // for ( int kfid = 0; kfid < kf_loop_id; kfid++ ) {
  //   auto kf = map_manager_->GetKeyframe( kfid );
  //   if ( kf == nullptr ) {
  //     continue;
  //   }
  //   Sophus::SE3d Twc = kf->GetTwc();
  //   map_id_poses_param_block.emplace( kfid, backend::PoseParametersBlock( kfid, Twc ) );
  // }
  // backend::examples::OutputPoses( "./initial_poses.txt", map_id_poses_param_block );

  tceres::Solver::Options options;
  options.linear_solver_type = tceres::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10;
  options.function_tolerance = 1.e-4;

  tceres::Solver::Summary summary;
  tceres::Solve( options, &problem, &summary );

  // backend::examples::OutputPoses( "./optimization_poses.txt", map_id_poses_param_block );

  Sophus::SE3d opt_newkf_pose = map_id_poses_param_block.at( new_frame.kfid_ ).getPose();
  if ( ( opt_newkf_pose.translation() - new_Twc.translation() ).norm() > 0.3 &&
       param_->slam_setting_.stereo_mode_ ) {
    LOG( WARNING ) << "[PoseGraph] Skipping as we are most likely with a degenerate solution!";
    return false;
  }

  std::unordered_set<int> processed_lm_ids;

  std::vector<int> vlm_ids, vkf_ids;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vwpt;
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vTwc;

  // get updated KFs / MPs
  for ( const auto& kf_id_pkf : map_kfs ) {
    int kf_id = kf_id_pkf.first;
    auto pkf = kf_id_pkf.second;

    if ( pkf == nullptr ) {
      continue;
    }

    vkf_ids.push_back( kf_id );
    vTwc.push_back( map_id_poses_param_block.at( kf_id ).getPose() );

    for ( const auto& kp : pkf->GetKeypoints3d() ) {
      int lmid = kp.lmid_;
      if ( processed_lm_ids.count( lmid ) ) {
        continue;
      }
      auto plm = map_manager_->GetMapPoint( lmid );
      if ( plm == nullptr ) {
        map_manager_->RemoveMapPointObs( lmid, kf_id );
        continue;
      }
      if ( plm->kfid_ == kf_id ) {
        Eigen::Vector3d cam_pt = pkf->ProjWorldToCam( plm->GetPoint() );
        Eigen::Vector3d new_wpt = vTwc.back() * cam_pt;

        vlm_ids.push_back( lmid );
        vwpt.push_back( new_wpt );
        processed_lm_ids.insert( lmid );
      }
    }
  }

  std::lock_guard<std::mutex> lck( map_manager_->map_mutex_ );

  Sophus::SE3d init_Tcw = new_frame.GetTcw();

  // Propagate corrections to youngest KFs / MPs
  int cur_kfs_num = map_manager_->GetNumberKF();
  for ( int kfid = new_frame.kfid_ + 1; kfid <= cur_kfs_num; kfid++ ) {
    auto pkf = map_manager_->GetKeyframe( kfid );
    if ( pkf == nullptr ) {
      continue;
    }

    Sophus::SE3d Tw_cur = pkf->GetTwc();
    Sophus::SE3d T_lc_cur = init_Tcw * Tw_cur;
    Sophus::SE3d update_T_wc_cur = opt_newkf_pose * T_lc_cur;

    for ( const auto& kp : pkf->GetKeypoints3d() ) {
      int lmid = kp.lmid_;
      if ( processed_lm_ids.count( lmid ) ) {
        continue;
      }
      auto plm = map_manager_->GetMapPoint( lmid );
      if ( plm == nullptr ) {
        map_manager_->RemoveMapPointObs( lmid, kfid );
        continue;
      }
      if ( plm->kfid_ == kfid ) {
        Eigen::Vector3d cam_pt = pkf->ProjWorldToCam( plm->GetPoint() );
        Eigen::Vector3d wpt = update_T_wc_cur * cam_pt;

        map_manager_->UpdateMapPoint( lmid, wpt );
        processed_lm_ids.insert( lmid );
      }
    }
    pkf->SetTwc( update_T_wc_cur );
  }

  // update mappoint
  for ( size_t i = 0; i < vlm_ids.size(); i++ ) {
    int lmid = vlm_ids.at( i );
    map_manager_->UpdateMapPoint( lmid, vwpt.at( i ) );
  }
  // update kf pose
  for ( size_t i = 0; i < vkf_ids.size(); i++ ) {
    int kf_id = vkf_ids.at( i );
    auto pkf = map_manager_->GetKeyframe( kf_id );
    if ( pkf != nullptr ) {
      pkf->SetTwc( vTwc.at( i ) );
    }
  }

  // update current frame pose
  Sophus::SE3d pre_Twc_cur = map_manager_->GetCurrentFrame()->GetTwc();
  Sophus::SE3d T_lc_curframe = init_Tcw * pre_Twc_cur;
  Sophus::SE3d update_Twcur = opt_newkf_pose * T_lc_curframe;
  map_manager_->GetCurrentFrame()->SetTwc( update_Twcur );

  return true;
}

void Optimization::LocalBA( viohw::Frame& newframe, bool buse_robust_cost ) {}

}  // namespace viohw
