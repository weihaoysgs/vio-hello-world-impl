// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "backend/cost_function/se3_left_pgo_factor.hpp"
#include "backend/example/common/read_g2o_format.hpp"
#include "backend/parameter_block/se3_parameter_block.hpp"
#include "backend/parameterization/se3left_parametrization.hpp"
#include "tceres/problem.h"
#include "tceres/solver.h"

DEFINE_string( input, "../backend/data/sphere.g2o",
               "The pose graph definition filename in g2o format." );

bool CheckParameterValueValid( double* param, int size ) {
  for ( int i = 0; i < size; i++ ) {
    if ( std::isnan( param[i] ) || std::isinf( param[i] ) ) {
      return false;
    }
  }
  return true;
}

// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem( const backend::examples::VectorOfConstraints& constraints,
                               backend::examples::MapOfPoses* poses, tceres::Problem* problem ) {
  CHECK( poses != NULL );
  CHECK( problem != NULL );

  CHECK( backend::examples::OutputPoses( "poses_original.txt", *poses ) )
      << "Error outputting to poses_original.txt";

  if ( constraints.empty() ) {
    LOG( INFO ) << "No constraints, no problem to optimize.";
    return;
  }
  std::map<int, backend::PoseParametersBlock> map_id_poses_params_;

  for ( const auto& pose : *poses ) {
    Sophus::SE3d T( pose.second.q, pose.second.p );
    map_id_poses_params_.emplace( pose.first, backend::PoseParametersBlock( pose.first, T ) );
    tceres::LocalParameterization* local_parameterization = new backend::SE3LeftParameterization();
    CHECK( CheckParameterValueValid( map_id_poses_params_.at( pose.first ).values(), 7 ) );
    problem->AddParameterBlock( map_id_poses_params_.at( pose.first ).values(), 7,
                                local_parameterization );
  }
  auto pose_start_iter = map_id_poses_params_.begin();
  CHECK( pose_start_iter != map_id_poses_params_.end() ) << "There are no poses.";
  problem->SetParameterBlockConstant( map_id_poses_params_.at( pose_start_iter->first ).values() );
  CHECK( map_id_poses_params_.size() == poses->size() ) << "pose param size not equal";

  for ( auto constraints_iter = constraints.begin(); constraints_iter != constraints.end();
        ++constraints_iter ) {
    const backend::examples::Constraint3d& constraint = *constraints_iter;

    auto pose_begin_iter = map_id_poses_params_.find( constraint.id_begin );
    auto pose_end_iter = map_id_poses_params_.find( constraint.id_end );
    CHECK( pose_begin_iter != map_id_poses_params_.end() );
    CHECK( pose_end_iter != map_id_poses_params_.end() );

    const Eigen::Matrix<double, 6, 6> sqrt_information = constraint.information.llt().matrixL();
    Sophus::SE3d Tb0b1( constraint.t_be.q, constraint.t_be.p );
    auto* cost_function = new backend::LeftSE3RelativePoseError( Tb0b1, sqrt_information );
    problem->AddResidualBlock( cost_function, nullptr,
                               map_id_poses_params_.at( pose_begin_iter->first ).values(),
                               map_id_poses_params_.at( pose_end_iter->first ).values() );
  }

  CHECK( problem != nullptr );

  tceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = tceres::SPARSE_NORMAL_CHOLESKY;
  options.check_gradients = true;
  options.gradient_check_relative_precision = 2.0;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = tceres::DOGLEG;

  tceres::Solver::Summary summary;
  tceres::Solve( options, problem, &summary );

  std::cout << summary.FullReport() << '\n';
  CHECK( backend::examples::OutputPoses( "poses_optimized.txt", map_id_poses_params_ ) )
      << "Error outputting to poses_original.txt";
}

// python3 plot_results.py --initial_poses ./poses_original.txt --optimized_poses\
// ./poses_optimized.txt

int main( int argc, char** argv ) {
  google::InitGoogleLogging( "pgo_test" );
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = google::INFO;
  google::ParseCommandLineFlags( &argc, &argv, true );
  backend::examples::MapOfPoses poses;
  backend::examples::VectorOfConstraints constraints;

  CHECK( !FLAGS_input.empty() ) << "Need to specify the filename to read.";
  CHECK( backend::examples::ReadG2oFile( FLAGS_input, &poses, &constraints ) )
      << "Error reading the file: " << FLAGS_input;

  LOG( INFO ) << "Number of poses: " << poses.size() << '\n';
  LOG( INFO ) << "Number of constraints: " << constraints.size() << '\n';

  tceres::Problem problem;
  BuildOptimizationProblem( constraints, &poses, &problem );

  return 0;
}