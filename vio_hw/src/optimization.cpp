#include "vio_hw/internal/optimization.hpp"

namespace viohw {
viohw::Optimization::Optimization( viohw::SettingPtr param, viohw::MapManagerPtr map_manager )
    : map_manager_( map_manager ), param_( param ) {}

bool Optimization::LocalPoseGraph( Frame &new_frame, int kf_loop_id, const Sophus::SE3d &new_Twc ) {
  tceres::Problem problem;

  std::map<int, backend::PoseParametersBlock> map_id_poses_param_block;
  std::map<int, std::shared_ptr<Frame>> map_kfs;

  auto loop_kf = map_manager_->GetKeyframe( kf_loop_id );
  CHECK( loop_kf != nullptr ) << "Loop KF empty";

  // add loop kf pose to problem and set constant
  Sophus::SE3d T_loop_wc = loop_kf->GetTwc();
  map_id_poses_param_block.emplace( kf_loop_id,
                                    backend::PoseParametersBlock( kf_loop_id, T_loop_wc ) );
  auto *local_pose_parameterization = new backend::SE3LeftParameterization;
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
    auto *local_parameterization = new backend::SE3LeftParameterization;
    problem.AddParameterBlock( map_id_poses_param_block.at( kfid ).values(),
                               backend::PoseParametersBlock::ndim_, local_parameterization );
    Sophus::SE3d T_ij = T_ci_w * T_w_cj;
    tceres::CostFunction *cost_function = new backend::LeftSE3RelativePoseError( T_ij );
    problem.AddResidualBlock( cost_function, nullptr,
                              map_id_poses_param_block.at( ci_kf_id ).values(),
                              map_id_poses_param_block.at( kfid ).values() );
    T_ci_w = T_w_cj.inverse();
    ci_kf_id = kfid;
  }

  tceres::CostFunction *cost_fun = new backend::LeftSE3RelativePoseError( T_loopkf_newkf );
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

  std::lock_guard<std::mutex> lock( map_manager_->map_mutex_ );
  // get updated KFs / MPs
  for ( const auto &kf_id_pkf : map_kfs ) {
    int kf_id = kf_id_pkf.first;
    auto pkf = kf_id_pkf.second;

    if ( pkf == nullptr ) {
      continue;
    }

    vkf_ids.push_back( kf_id );
    vTwc.push_back( map_id_poses_param_block.at( kf_id ).getPose() );

    for ( const auto &kp : pkf->GetKeypoints3d() ) {
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

    for ( const auto &kp : pkf->GetKeypoints3d() ) {
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

void Optimization::LocalBA( Frame &newframe, const bool buse_robust_cost ) {
  using namespace backend;

  std::vector<std::shared_ptr<Frame>> inertial_frames;
  std::vector<std::shared_ptr<Frame>> fix_frames;

  std::shared_ptr<Frame> curr_frame = map_manager_->GetKeyframe( newframe.kfid_ );
  assert( curr_frame != nullptr );
  inertial_frames.push_back( curr_frame );
  const int opt_kf_num = param_->backend_optimization_setting_.optimize_kf_num_;
  for ( int i = 1; i <= opt_kf_num; i++ ) {
    int last_kf_id = newframe.kfid_ - i;
    std::shared_ptr<Frame> last_kf = map_manager_->GetKeyframe( last_kf_id );
    if ( last_kf_id < 0 || last_kf == nullptr ) {
      continue;
    }
    inertial_frames.push_back( last_kf );
  }
  std::sort( inertial_frames.begin(), inertial_frames.end(),
             []( const std::shared_ptr<Frame> f1, const std::shared_ptr<Frame> f2 ) {
               return f1->kfid_ > f2->kfid_;
             } );
  // =================================
  //      Setup BA Problem
  // =================================

  tceres::Problem problem;
  tceres::LossFunctionWrapper *loss_function;

  // Chi2 thresh.
  const float mono_th = 5.991;

  loss_function = new tceres::LossFunctionWrapper( new tceres::HuberLoss( std::sqrt( mono_th ) ),
                                                   tceres::TAKE_OWNERSHIP );

  if ( !buse_robust_cost ) {
    loss_function->Reset( nullptr, tceres::TAKE_OWNERSHIP );
  }

  size_t nmincstkfs = 2;

  if ( param_->slam_setting_.stereo_mode_ ) {
    nmincstkfs = 1;
  }

  size_t nbmono = 0;
  size_t nbstereo = 0;

  auto ordering = new tceres::ParameterBlockOrdering;

  std::unordered_map<int, PoseParametersBlock> map_id_posespar_;
  std::unordered_map<int, PointXYZParametersBlock> map_id_pointspar_;

  std::unordered_map<int, std::shared_ptr<MapPoint>> map_local_plms;
  std::unordered_map<int, std::shared_ptr<Frame>> map_local_pkfs;

  // Storing the factors and their residuals block ids
  // for fast accessing when cheking for outliers
  std::vector<
      std::pair<tceres::CostFunction *, std::pair<tceres::ResidualBlockId, std::pair<int, int>>>>
      vreprojerr_kfid_lmid, vright_reprojerr_kfid_lmid, vanchright_reprojerr_kfid_lmid;

  // Add the left cam calib parameters
  auto pcalibleft = newframe.pcalib_leftcam_;
  CameraIntrinsicParametersBlock calibpar( 0, pcalibleft->fx_, pcalibleft->fy_, pcalibleft->cx_,
                                           pcalibleft->cy_ );
  problem.AddParameterBlock( calibpar.values(), 4 );
  ordering->AddElementToGroup( calibpar.values(), 1 );

  problem.SetParameterBlockConstant( calibpar.values() );

  // Prepare variables if STEREO mode
  auto pcalibright = newframe.pcalib_rightcam_;
  CameraIntrinsicParametersBlock rightcalibpar;

  Sophus::SE3d Trl, Tlr;
  PoseParametersBlock rlextrinpose( 0, Trl );

  if ( param_->slam_setting_.stereo_mode_ ) {
    // Right Intrinsic
    rightcalibpar = CameraIntrinsicParametersBlock( 0, pcalibright->fx_, pcalibright->fy_,
                                                    pcalibright->cx_, pcalibright->cy_ );
    problem.AddParameterBlock( rightcalibpar.values(), 4 );
    ordering->AddElementToGroup( rightcalibpar.values(), 1 );

    problem.SetParameterBlockConstant( rightcalibpar.values() );

    // Right Extrinsic
    Tlr = pcalibright->getExtrinsic();
    Trl = Tlr.inverse();
    rlextrinpose = PoseParametersBlock( 0, Trl );

    tceres::LocalParameterization *local_param = new SE3LeftParameterization();

    problem.AddParameterBlock( rlextrinpose.values(), 7, local_param );
    ordering->AddElementToGroup( rlextrinpose.values(), 1 );

    problem.SetParameterBlockConstant( rlextrinpose.values() );
  }

  // Keep track of MPs no suited for BA for speed-up
  std::unordered_set<int> set_badlmids;

  std::unordered_set<int> set_lmids2opt;
  std::unordered_set<int> set_kfids2opt;
  std::unordered_set<int> set_cstkfids;

  for ( int i = 0; i < inertial_frames.size(); i++ ) {
    std::shared_ptr<Frame> pkf = inertial_frames[i];

    map_id_posespar_.emplace( pkf->kfid_, PoseParametersBlock( pkf->kfid_, pkf->GetTwc() ) );
    tceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();
    problem.AddParameterBlock( map_id_posespar_.at( pkf->kfid_ ).values(), 7,
                               local_parameterization );
    ordering->AddElementToGroup( map_id_posespar_.at( pkf->kfid_ ).values(), 1 );

    // Get the 3D point for those keyframes, latter deal
    set_kfids2opt.insert( pkf->kfid_ );
    for ( const auto &kp : pkf->GetKeypoints3d() ) {
      set_lmids2opt.insert( kp.lmid_ );
    }
    map_local_pkfs.emplace( pkf->kfid_, pkf );
  }

  // set const frame
  {
    std::shared_ptr<Frame> before_kf = inertial_frames.back();
    problem.SetParameterBlockConstant( map_id_posespar_.at( before_kf->kfid_ ).values() );
    if ( nmincstkfs > 1 ) {
      std::shared_ptr<Frame> kf = inertial_frames.at( inertial_frames.size() - 1 );
      problem.SetParameterBlockConstant( map_id_posespar_.at( kf->kfid_ ).values() );
    }
  }

  // Go through the MPs to optimize
  for ( const auto &lmid : set_lmids2opt ) {
    auto plm = map_manager_->GetMapPoint( lmid );

    if ( plm == nullptr ) {
      continue;
    }

    if ( plm->IsBad() ) {
      set_badlmids.insert( lmid );
      continue;
    }

    map_local_plms.emplace( lmid, plm );

    map_id_pointspar_.emplace( lmid, PointXYZParametersBlock( lmid, plm->GetPoint() ) );

    problem.AddParameterBlock( map_id_pointspar_.at( lmid ).values(), 3 );
    ordering->AddElementToGroup( map_id_pointspar_.at( lmid ).values(), 0 );

    for ( const auto &kfid : plm->GetKfObsSet() ) {
      auto pkfit = map_local_pkfs.find( kfid );
      std::shared_ptr<Frame> pkf = nullptr;

      // Add the observing KF if not set yet
      if ( pkfit == map_local_pkfs.end() ) {
        pkf = map_manager_->GetKeyframe( kfid );
        if ( pkf == nullptr ) {
          map_manager_->RemoveMapPointObs( kfid, plm->lmid_ );
          continue;
        }
        map_local_pkfs.emplace( kfid, pkf );
        map_id_posespar_.emplace( kfid, PoseParametersBlock( kfid, pkf->GetTwc() ) );

        tceres::LocalParameterization *local_parameterization = new SE3LeftParameterization();

        problem.AddParameterBlock( map_id_posespar_.at( kfid ).values(), 7,
                                   local_parameterization );
        ordering->AddElementToGroup( map_id_posespar_.at( kfid ).values(), 1 );

        set_cstkfids.insert( kfid );
        problem.SetParameterBlockConstant( map_id_posespar_.at( kfid ).values() );

      } else {
        pkf = pkfit->second;
      }

      auto kp = pkf->GetKeypointById( lmid );

      if ( kp.lmid_ != lmid ) {
        map_manager_->RemoveMapPointObs( lmid, kfid );
        continue;
      }

      tceres::CostFunction *f;
      tceres::ResidualBlockId rid;

      // Add a visual factor between KF-MP nodes
      if ( kp.is_stereo_ ) {
        f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ( kp.unpx_.x, kp.unpx_.y,
                                                         std::pow( 2., kp.scale_ ) );

        rid = problem.AddResidualBlock( f, loss_function, calibpar.values(),
                                        map_id_posespar_.at( kfid ).values(),
                                        map_id_pointspar_.at( lmid ).values() );

        vreprojerr_kfid_lmid.push_back(
            std::make_pair( f, std::make_pair( rid, std::make_pair( kfid, lmid ) ) ) );

        f = new DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ( kp.runpx_.x, kp.runpx_.y,
                                                                 std::pow( 2., kp.scale_ ) );

        rid = problem.AddResidualBlock( f, loss_function, rightcalibpar.values(),
                                        map_id_posespar_.at( kfid ).values(), rlextrinpose.values(),
                                        map_id_pointspar_.at( lmid ).values() );

        vright_reprojerr_kfid_lmid.push_back(
            std::make_pair( f, std::make_pair( rid, std::make_pair( kfid, lmid ) ) ) );
        nbstereo++;
        continue;
      } else {
        f = new DirectLeftSE3::ReprojectionErrorKSE3XYZ( kp.unpx_.x, kp.unpx_.y,
                                                         std::pow( 2., kp.scale_ ) );

        rid = problem.AddResidualBlock( f, loss_function, calibpar.values(),
                                        map_id_posespar_.at( kfid ).values(),
                                        map_id_pointspar_.at( lmid ).values() );

        vreprojerr_kfid_lmid.push_back(
            std::make_pair( f, std::make_pair( rid, std::make_pair( kfid, kp.lmid_ ) ) ) );

        nbmono++;
      }
    }
  }

  // Ensure the gauge is fixed
  size_t nbcstkfs = set_cstkfids.size();

  // At least two fixed KF in mono / one fixed KF in Stereo / RGB-D
  if ( nbcstkfs < nmincstkfs ) {
    for ( auto it = map_local_pkfs.begin(); nbcstkfs < nmincstkfs && it != map_local_pkfs.end();
          it++ ) {
      problem.SetParameterBlockConstant( map_id_posespar_.at( it->first ).values() );
      set_cstkfids.insert( it->first );
      nbcstkfs++;
    }
  }

  size_t nbkfstot = map_local_pkfs.size();
  size_t nbkfs2opt = nbkfstot - nbcstkfs;
  size_t nblms2opt = map_local_plms.size();

  // =================================
  //      Solve BA Problem
  // =================================

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

  // =================================
  //      Remove outliers
  // =================================

  size_t nbbadobsmono = 0;
  size_t nbbadobsrightcam = 0;

  std::vector<std::pair<int, int>> vbadkflmids;
  std::vector<std::pair<int, int>> vbadstereokflmids;
  vbadkflmids.reserve( vreprojerr_kfid_lmid.size() / 10 );
  vbadstereokflmids.reserve( vright_reprojerr_kfid_lmid.size() / 10 );

  for ( auto it = vreprojerr_kfid_lmid.begin(); it != vreprojerr_kfid_lmid.end(); ) {
    bool bbigchi2 = true;
    bool bdepthpos = true;

    auto *err = static_cast<DirectLeftSE3::ReprojectionErrorKSE3XYZ *>( it->first );
    bbigchi2 = err->chi2err_ > mono_th;
    bdepthpos = err->isdepthpositive_;

    if ( bbigchi2 || !bdepthpos ) {
      int lmid = it->second.second.second;
      int kfid = it->second.second.first;
      vbadkflmids.push_back( std::pair<int, int>( kfid, lmid ) );
      set_badlmids.insert( lmid );
      nbbadobsmono++;

      it = vreprojerr_kfid_lmid.erase( it );
    } else {
      it++;
    }
  }

  for ( auto it = vright_reprojerr_kfid_lmid.begin(); it != vright_reprojerr_kfid_lmid.end(); ) {
    bool bbigchi2 = true;
    bool bdepthpos = true;
    auto *err = static_cast<DirectLeftSE3::ReprojectionErrorRightCamKSE3XYZ *>( it->first );
    bbigchi2 = err->chi2err_ > mono_th;
    bdepthpos = err->isdepthpositive_;
    if ( bbigchi2 || !bdepthpos ) {
      int lmid = it->second.second.second;
      int kfid = it->second.second.first;
      vbadstereokflmids.push_back( std::pair<int, int>( kfid, lmid ) );
      set_badlmids.insert( lmid );
      nbbadobsrightcam++;

      it = vright_reprojerr_kfid_lmid.erase( it );
    } else {
      it++;
    }
  }

  size_t nbbadobs = nbbadobsmono + nbbadobsrightcam;

  // =================================
  //      Refine BA Solution
  // =================================

  bool bl2optimdone = false;

  // =================================
  //      Remove outliers
  // =================================

  // =================================
  //      Update State Parameters
  // =================================

  std::lock_guard<std::mutex> lock( map_manager_->map_mutex_ );

  for ( const auto &badkflmid : vbadstereokflmids ) {
    int kfid = badkflmid.first;
    int lmid = badkflmid.second;
    auto it = map_local_pkfs.find( kfid );
    if ( it != map_local_pkfs.end() ) {
      it->second->RemoveStereoKeypointById( lmid );
    }
    set_badlmids.insert( lmid );
  }

  for ( const auto &badkflmid : vbadkflmids ) {
    int kfid = badkflmid.first;
    int lmid = badkflmid.second;
    auto it = map_local_pkfs.find( kfid );
    if ( it != map_local_pkfs.end() ) {
      map_manager_->RemoveMapPointObs( lmid, kfid );
    }
    if ( kfid == map_manager_->GetCurrentFrame()->kfid_ ) {
      map_manager_->RemoveObsFromCurFrameById( lmid );
    }
    set_badlmids.insert( lmid );
  }

  // Update KFs
  for ( const auto &kfid_pkf : map_local_pkfs ) {
    int kfid = kfid_pkf.first;

    if ( set_cstkfids.count( kfid ) ) {
      continue;
    }

    auto pkf = kfid_pkf.second;

    if ( pkf == nullptr ) {
      continue;
    }

    // auto optkfpose = map_id_posespar_.at(kfid);
    auto it = map_id_posespar_.find( kfid );
    if ( it != map_id_posespar_.end() ) {
      pkf->SetTwc( it->second.getPose() );
    }
  }

  // Update MPs
  for ( const auto &lmid_plm : map_local_plms ) {
    int lmid = lmid_plm.first;
    auto plm = lmid_plm.second;

    if ( plm == nullptr ) {
      set_badlmids.erase( lmid );
      continue;
    }

    if ( plm->IsBad() ) {
      map_manager_->RemoveMapPoint( lmid );
      set_badlmids.erase( lmid );
      continue;
    }

    // Map Point Culling
    auto kfids = plm->GetKfObsSet();
    if ( kfids.size() < 3 ) {
      if ( plm->kfid_ < newframe.kfid_ - 3 && !plm->isobs_ ) {
        map_manager_->RemoveMapPoint( lmid );
        set_badlmids.erase( lmid );
        continue;
      }
    }

    auto optlmit = map_id_pointspar_.find( lmid );
    if ( optlmit != map_id_pointspar_.end() ) {
      map_manager_->UpdateMapPoint( lmid, optlmit->second.getPoint() );
    } else {
      set_badlmids.insert( lmid );
    }
  }

  // Map Point Culling for bad Obs.
  size_t nbbadlm = 0;
  for ( const auto &lmid : set_badlmids ) {
    std::shared_ptr<MapPoint> plm;
    auto plmit = map_local_plms.find( lmid );
    if ( plmit == map_local_plms.end() ) {
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

  nbbadobs = nbbadobsmono + nbbadobsrightcam;
}

}  // namespace viohw
