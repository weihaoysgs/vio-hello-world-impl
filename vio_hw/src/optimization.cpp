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

void Optimization::LocalBA( Frame& newframe, bool use_robust_cost ) {
  const int optimize_frame_num = 25;
  const float mono_th = 5.9915;
  const int min_cov_score = 20;
  size_t min_const_kfs = 2;
  size_t num_mono = 0;
  size_t num_stereo = 0;

  std::vector<std::shared_ptr<Frame>> optimized_frame;
  std::shared_ptr<Frame> curr_frame = map_manager_->GetKeyframe( newframe.kfid_ );
  CHECK( curr_frame != nullptr ) << "Opt KF is nullptr";
  optimized_frame.push_back( curr_frame );

  // push back to be optimized keyframe
  for ( int i = 1; i <= optimize_frame_num; i++ ) {
    int last_kf_id = curr_frame->kfid_ - i;
    auto last_kf = map_manager_->GetKeyframe( last_kf_id );
    if ( last_kf == nullptr || last_kf_id < 0 ) {
      continue;
    }
    optimized_frame.push_back( last_kf );
  }

  // from big to small
  std::sort( optimized_frame.begin(), optimized_frame.end(),
             []( const std::shared_ptr<Frame>& f1, const std::shared_ptr<Frame>& f2 ) {
               return f1->kfid_ > f2->kfid_;
             } );

  //==========Setup BA Problem==========//

  tceres::Problem problem;
  auto* loss_function = new tceres::LossFunctionWrapper(
      new tceres::HuberLoss( std::sqrt( mono_th ) ), tceres::TAKE_OWNERSHIP );

  if ( !use_robust_cost ) {
    loss_function->Reset( nullptr, tceres::TAKE_OWNERSHIP );
  }

  if ( curr_frame->nb3dkps_ < min_cov_score ) {
    LOG( WARNING ) << "Current KF kps too less than " << min_cov_score << ", not execute LocalBA()";
    return;
  }

  if ( param_->slam_setting_.stereo_mode_ ) {
    min_const_kfs = 1;
  }

  auto ordering = new tceres::ParameterBlockOrdering;

  std::unordered_map<int, backend::PoseParametersBlock> map_id_pose_params;
  std::unordered_map<int, backend::PointXYZParametersBlock> map_id_point_xyz_params;

  std::unordered_map<int, std::shared_ptr<Frame>> map_local_kfs;
  std::unordered_map<int, std::shared_ptr<MapPoint>> map_local_landmarks;

  // Storing the factors and their residuals block ids
  // for fast accessing when checking for outliers
  std::vector<std::pair<tceres::CostFunction*,
                        std::pair<tceres::internal::ResidualBlock*, std::pair<int, int>>>>
      v_reproj_err_kfid_lmid, v_right_reproj_err_kfid_lmid;

  //==========Setup Intrinsic and Extrinsic parameters==========//

  // add left camera intrinsic parameter to problem and set constant
  auto left_cam_calib = newframe.pcalib_leftcam_;
  backend::CameraIntrinsicParametersBlock left_cam_intrinsic_param_block( 0, left_cam_calib->K_ );
  problem.AddParameterBlock( left_cam_intrinsic_param_block.values(), 4 );
  ordering->AddElementToGroup( left_cam_intrinsic_param_block.values(), 1 );
  problem.SetParameterBlockConstant( left_cam_intrinsic_param_block.values() );

  // prepare variable for stereo mode
  auto right_cam_calib = newframe.pcalib_rightcam_;
  backend::CameraIntrinsicParametersBlock right_cam_intrinsic_param_block;
  Sophus::SE3d Trl, Tlr;
  backend::PoseParametersBlock rl_extrinsic_pose_parameter_block( 0, Trl );

  if ( param_->slam_setting_.stereo_mode_ ) {
    // right Intrinsic
    right_cam_intrinsic_param_block =
        backend::CameraIntrinsicParametersBlock( 0, right_cam_calib->K_ );
    problem.AddParameterBlock( right_cam_intrinsic_param_block.values(), 4 );
    ordering->AddElementToGroup( right_cam_intrinsic_param_block.values(), 1 );
    problem.SetParameterBlockConstant( right_cam_intrinsic_param_block.values() );

    // right Extrinsic
    Tlr = right_cam_calib->getExtrinsic();
    Trl = Tlr.inverse();
    rl_extrinsic_pose_parameter_block = backend::PoseParametersBlock( 0, Trl );
    tceres::LocalParameterization* local_param = new backend::SE3LeftParameterization();
    problem.AddParameterBlock( rl_extrinsic_pose_parameter_block.values(), 7, local_param );
    ordering->AddElementToGroup( rl_extrinsic_pose_parameter_block.values(), 1 );
    problem.SetParameterBlockConstant( rl_extrinsic_pose_parameter_block.values() );
  }

  //==========Setup Tobe Optimized (KF Pose && Point) parameters==========//

  std::unordered_set<int> set_badlmids;
  std::unordered_set<int> set_lmids2opt;
  std::unordered_set<int> set_kfids2opt;
  std::unordered_set<int> set_cstkfids;

  for ( int i = 0; i < optimized_frame.size(); i++ ) {
    std::shared_ptr<Frame> pkf = optimized_frame[i];
    map_id_pose_params.emplace( pkf->kfid_,
                                backend::PoseParametersBlock( pkf->kfid_, pkf->GetTwc() ) );
    tceres::LocalParameterization* local_parameterization = new backend::SE3LeftParameterization();
    problem.AddParameterBlock( map_id_pose_params.at( pkf->kfid_ ).values(), 7,
                               local_parameterization );
    ordering->AddElementToGroup( map_id_pose_params.at( pkf->kfid_ ).values(), 1 );

    // set first frame constant
    if ( i == optimized_frame.size() - 1 ) {
      set_cstkfids.insert( pkf->kfid_ );
      problem.SetParameterBlockConstant( map_id_pose_params.at( pkf->kfid_ ).values() );
    }

    // Get the 3D point for those keyframes, latter deal
    set_kfids2opt.insert( pkf->kfid_ );
    for ( const auto& kp : pkf->GetKeypoints3d() ) {
      set_lmids2opt.insert( kp.lmid_ );
    }
    map_local_kfs.emplace( pkf->kfid_, pkf );
  }


  // Go through the MPs to optimize
  for ( const auto& lmid : set_lmids2opt ) {
    auto plm = map_manager_->GetMapPoint( lmid );

    if ( plm == nullptr ) {
      continue;
    }

    if ( plm->IsBad() ) {
      set_badlmids.insert( lmid );
      continue;
    }

    map_local_landmarks.emplace( lmid, plm );

    map_id_point_xyz_params.emplace( lmid,
                                     backend::PointXYZParametersBlock( lmid, plm->GetPoint() ) );
    problem.AddParameterBlock( map_id_point_xyz_params.at( lmid ).values(), 3 );
    ordering->AddElementToGroup( map_id_point_xyz_params.at( lmid ).values(), 0 );

    for ( const auto& kfid : plm->GetKfObsSet() ) {
      // if ( !set_kfids2opt.count( kfid ) && !set_cstkfids.count( kfid ) ) {
      //   continue;
      // }
      auto kf_iter = map_local_kfs.find( kfid );
      std::shared_ptr<Frame> pkf = nullptr;
      if ( kf_iter == map_local_kfs.end() ) {
        pkf = map_manager_->GetKeyframe( kfid );
        if ( pkf == nullptr ) {
          map_manager_->RemoveMapPointObs( kfid, plm->lmid_ );
          continue;
        }

        map_local_kfs.emplace( kfid, pkf );
        map_id_pose_params.emplace( kfid, backend::PoseParametersBlock( kfid, pkf->GetTwc() ) );

        tceres::LocalParameterization* local_parameterization =
            new backend::SE3LeftParameterization();
        problem.AddParameterBlock( map_id_pose_params.at( kfid ).values(), 7,
                                   local_parameterization );
        ordering->AddElementToGroup( map_id_pose_params.at( kfid ).values(), 1 );
        problem.SetParameterBlockConstant( map_id_pose_params.at( kfid ).values() );

        set_cstkfids.insert( kfid );
      } else {
        pkf = kf_iter->second;
      }

      auto kp = pkf->GetKeypointById( lmid );
      // if not found, the id is -1
      if ( kp.lmid_ != lmid ) {
        map_manager_->RemoveMapPointObs( lmid, kfid );
        continue;
      }

      tceres::CostFunction* f;
      tceres::ResidualBlockId rid;

      if ( kp.is_stereo_ ) {
        f = new backend::DirectLeftSE3::ReprojectionErrorKSE3XYZ( kp.unpx_.x, kp.unpx_.y,
                                                                  std::pow( 2., kp.scale_ ) );

        rid = problem.AddResidualBlock( f, loss_function, left_cam_intrinsic_param_block.values(),
                                        map_id_pose_params.at( kfid ).values(),
                                        map_id_point_xyz_params.at( lmid ).values() );

        v_reproj_err_kfid_lmid.push_back(
            std::make_pair( f, std::make_pair( rid, std::make_pair( kfid, lmid ) ) ) );

        f = new backend::DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ(
            kp.runpx_.x, kp.runpx_.y, std::pow( 2., kp.scale_ ) );

        rid = problem.AddResidualBlock( f, loss_function, right_cam_intrinsic_param_block.values(),
                                        map_id_pose_params.at( kfid ).values(),
                                        rl_extrinsic_pose_parameter_block.values(),
                                        map_id_point_xyz_params.at( lmid ).values() );

        v_right_reproj_err_kfid_lmid.push_back(
            std::make_pair( f, std::make_pair( rid, std::make_pair( kfid, lmid ) ) ) );
        num_stereo++;
        continue;
      } else {
        f = new backend::DirectLeftSE3::ReprojectionErrorKSE3XYZ( kp.unpx_.x, kp.unpx_.y,
                                                                  std::pow( 2., kp.scale_ ) );

        rid = problem.AddResidualBlock( f, loss_function, left_cam_intrinsic_param_block.values(),
                                        map_id_pose_params.at( kfid ).values(),
                                        map_id_point_xyz_params.at( lmid ).values() );
        v_reproj_err_kfid_lmid.push_back(
            std::make_pair( f, std::make_pair( rid, std::make_pair( kfid, kp.lmid_ ) ) ) );

        num_mono++;
      }
    }
  }

  tceres::Solver::Options options;
  options.linear_solver_ordering.reset( ordering );
  options.linear_solver_type = tceres::DENSE_SCHUR;
  options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;
  options.num_threads = 1;
  options.max_num_iterations = 5;
  options.function_tolerance = 1.e-3;
  options.max_solver_time_in_seconds = 0.2;

  tceres::Solver::Summary summary;
  tceres::Solve( options, &problem, &summary );


  // LOG( INFO ) << tceres::internal::StringPrintf( "constant kf %d, opt kf %d, opt landmarks %d",
  //                                                set_cstkfids.size(), set_kfids2opt.size(),
  //                                                set_lmids2opt.size() );

  // LOG( INFO ) << summary.FullReport();

  size_t num_bad_obs_mono = 0;
  size_t num_bad_obs_rightcam = 0;

  std::vector<std::pair<int, int>> v_bad_kflmids;
  std::vector<std::pair<int, int>> v_bad_stereo_kflmids;

  for ( auto it = v_reproj_err_kfid_lmid.begin(); it != v_reproj_err_kfid_lmid.end(); ) {
    auto* err = dynamic_cast<backend::DirectLeftSE3::ReprojectionErrorKSE3XYZ*>( it->first );
    bool big_chi2 = err->chi2err_ > mono_th;
    bool depth_positive = err->isdepthpositive_;
    if ( big_chi2 || !depth_positive ) {
      // TODO apply_l2_after_robust_

      int lmid = it->second.second.second;
      int kfid = it->second.second.first;
      v_bad_kflmids.push_back( std::pair<int, int>( kfid, lmid ) );
      set_badlmids.insert( lmid );
      num_bad_obs_mono++;

      it = v_reproj_err_kfid_lmid.erase( it );
    } else {
      it++;
    }
  }

  for ( auto it = v_right_reproj_err_kfid_lmid.begin();
        it != v_right_reproj_err_kfid_lmid.end(); ) {
    auto* err =
        dynamic_cast<backend::DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ*>( it->first );
    bool big_chi2 = err->chi2err_ > mono_th;
    bool depth_positive = err->isdepthpositive_;
    if ( big_chi2 || !depth_positive ) {
      // TODO apply_l2_after_robust_

      int lmid = it->second.second.second;
      int kfid = it->second.second.first;
      v_bad_stereo_kflmids.push_back( std::pair<int, int>( kfid, lmid ) );
      set_badlmids.insert( lmid );
      num_bad_obs_rightcam++;

      it = v_right_reproj_err_kfid_lmid.erase( it );
    } else {
      it++;
    }
  }

  // =================================
  //      Update State Parameters
  // =================================
  std::lock_guard<std::mutex> lock( map_manager_->map_mutex_ );
  for ( const auto& badkflmid : v_bad_stereo_kflmids ) {
    int kfid = badkflmid.first;
    int lmid = badkflmid.second;
    auto it = map_local_kfs.find( kfid );
    if ( it != map_local_kfs.end() ) {
      it->second->RemoveStereoKeypointById( lmid );
    }
    set_badlmids.insert( lmid );
  }

  for ( const auto& badkflmid : v_bad_kflmids ) {
    int kfid = badkflmid.first;
    int lmid = badkflmid.second;
    auto it = map_local_kfs.find( kfid );
    if ( it != map_local_kfs.end() ) {
      map_manager_->RemoveMapPointObs( lmid, kfid );
    }
    if ( kfid == map_manager_->GetCurrentFrame()->kfid_ ) {
      map_manager_->RemoveObsFromCurFrameById( lmid );
    }
    set_badlmids.insert( lmid );
  }

  // update KF pose
  for ( const auto& kfid_pkf : map_local_kfs ) {
    int kfid = kfid_pkf.first;

    if ( set_cstkfids.count( kfid ) ) {
      continue;
    }

    auto pkf = kfid_pkf.second;

    if ( pkf == nullptr ) {
      continue;
    }

    auto it = map_id_pose_params.find( kfid );
    if ( it != map_id_pose_params.end() ) {
      pkf->SetTwc( it->second.getPose() );
    }
  }

  LOG(INFO) << "Local BA end";
  // update mappoint
  /*for ( const auto& lmid_plm : map_local_landmarks ) {
    int lmid = lmid_plm.first;
    auto plm = lmid_plm.second;

    if ( plm == nullptr ) {
      set_badlmids.erase( lmid );
      continue;
    }

    if( plm->IsBad() ) {
      map_manager_->RemoveMapPoint(lmid);
      set_badlmids.erase(lmid);
      continue;
    }

    // Map Point Culling
    auto kfids = plm->GetKfObsSet();
    if( kfids.size() < 3 ) {
      if( plm->kfid_ < newframe.kfid_-3 && !plm->isobs_ ) {
        map_manager_->RemoveMapPoint(lmid);
        set_badlmids.erase(lmid);
        continue;
      }
    }

    auto optlmit = map_id_point_xyz_params.find( lmid );
    if ( optlmit != map_id_point_xyz_params.end() ) {
      map_manager_->UpdateMapPoint( lmid, optlmit->second.getPoint() );
    } else {
      set_badlmids.insert( lmid );
    }
  }

  // Map Point Culling for bad Obs.
  size_t nbbadlm = 0;
  for ( const auto& lmid : set_badlmids ) {
    std::shared_ptr<MapPoint> plm;
    auto plmit = map_local_landmarks.find( lmid );
    if ( plmit == map_local_landmarks.end() ) {
      plm = map_manager_->GetMapPoint( lmid );
    } else {
      plm = plmit->second;
    }
    if ( plm == nullptr ) {
      continue;
    }

    if ( plm->IsBad() ) {
      map_manager_->RemoveMapPoint( lmid );
      nbbadlm++;
    } else {
      auto set_cokfs = plm->GetKfObsSet();
      if ( set_cokfs.size() < 3 ) {
        if ( plm->kfid_ < newframe.kfid_ - 3 && !plm->isobs_ ) {
          map_manager_->RemoveMapPoint( lmid );
          nbbadlm++;
        }
      }
    }
  }
*/
}

}  // namespace viohw
