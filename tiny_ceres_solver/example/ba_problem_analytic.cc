/// A simple example of using the Ceres minimizer.
/// This file is part of VIO-Hello-World
/// Copyright (C) 2023 ZJU
/// You should have received a copy of the GNU General Public License,
/// see <https://www.gnu.org/licenses/> for more details
/// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include <utility>

#include "gflags/gflags.h"
#include "iostream"
#include "memory"
#include "random"
#include "tceres/parameter_block.h"
#include "tceres/problem.h"
#include "tceres/sized_cost_function.h"
#include "tceres/slam/inv_depth_parameter_block.h"
#include "tceres/slam/pose_local_parameterization.h"
#include "tceres/slam/pose_parameter_block.h"
#include "tceres/solver.h"
#include "tceres/stringprintf.h"
#include "unordered_map"

DEFINE_int32(pose_number, 3, "camera pose number");
DEFINE_int32(feature_num_per_frame, 20,
             "feature number observed by every camera");
DEFINE_double(radius, 8.0, "rotation radius");

class BundleAdjustmentCostFunction
    : public tceres::SizedCostFunction<2, 7, 7, 1>
{
 public:
  explicit BundleAdjustmentCostFunction(Eigen::Vector3d pt_i,
                                        Eigen::Vector3d pt_j)
      : obs_pt_i_(std::move(pt_i)), obs_pt_j_(std::move(pt_j)) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    // Twc
    Eigen::Map<const Eigen::Vector3d> Pi(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> Qi(parameters[0] + 3);

    Eigen::Map<const Eigen::Vector3d> Pj(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> Qj(parameters[1] + 3);

    double inverse_depth_i = parameters[2][0];

    Eigen::Vector3d pts_camera_i = obs_pt_i_ / inverse_depth_i;
    Eigen::Vector3d pts_world = Qi * pts_camera_i + Pi;
    Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_world - Pj);

    Eigen::Map<Eigen::Vector2d> error(residuals);
    double depth_j = pts_camera_j.z();
    error = (pts_camera_j / depth_j).head<2>() - obs_pt_j_.head<2>();

    if (jacobians) {
      using namespace tceres::slam;
      Eigen::Matrix3d Ri = Qi.toRotationMatrix();
      Eigen::Matrix3d Rj = Qj.toRotationMatrix();
      Eigen::Matrix<double, 2, 3> reduce(2, 3);
      reduce << 1. / depth_j, 0, -pts_camera_j(0) / (depth_j * depth_j), 0,
          1. / depth_j, -pts_camera_j(1) / (depth_j * depth_j);

      if (jacobians[0]) {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
            jacobians_pose_i(jacobians[0]);
        Eigen::Matrix<double, 3, 6> jacobians_pose;
        jacobians_pose.leftCols<3>() = Rj.transpose();
        jacobians_pose.rightCols<3>() =
            -Rj.transpose() * Ri * skewSymmetric(pts_camera_i);
        jacobians_pose_i.leftCols<6>() = reduce * jacobians_pose;
        jacobians_pose_i.rightCols<1>().setZero();
        {
          std::cout << "analysis jacobi pose i:\n"
                    << jacobians_pose_i << std::endl;
          const double eps = 1e-6;
          Eigen::Quaternion<double> q_temp;
          Eigen::Vector3d t_temp;
          Eigen::Matrix<double, 2, 6> numeric_jacobian;
          for (int i = 0; i < 6; i++) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(i % 3) = eps;
            if (i <= 2) {
              q_temp = Qi;
              t_temp = Pi + delta;
            } else {
              Eigen::Matrix<double, 3, 1> half_theta = delta;
              half_theta /= static_cast<double>(2.0);
              Eigen::Quaternion<double> dq(1.0, half_theta.x(), half_theta.y(),
                                           half_theta.z());
              q_temp = (Qi * dq);
              t_temp = Pi;
            }
            Eigen::Vector3d pts_camera_i = obs_pt_i_ / inverse_depth_i;
            Eigen::Vector3d pts_world = q_temp * pts_camera_i + t_temp;
            Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_world - Pj);
            Eigen::Vector2d _error;
            double depth_j = pts_camera_j.z();
            _error = (pts_camera_j / depth_j).head<2>() - obs_pt_j_.head<2>();
            numeric_jacobian.col(i) = (_error - error) / eps;
          }
          std::cout << "numeric jacobi pose i:\n " << numeric_jacobian
                    << std::endl;
        }
      }
      if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
            jacobians_pose_j(jacobians[1]);
        Eigen::Matrix<double, 3, 6> jacobians_pose;
        jacobians_pose.leftCols<3>() = -Rj.transpose();
        jacobians_pose.rightCols<3>() = skewSymmetric(pts_camera_j);
        jacobians_pose_j.leftCols<6>() = reduce * jacobians_pose;
        jacobians_pose_j.rightCols<1>().setZero();
        {
          std::cout << "analysis jacobi pose j:\n"
                    << jacobians_pose_j << std::endl;
          const double eps = 1e-6;
          Eigen::Quaternion<double> q_temp;
          Eigen::Vector3d t_temp;
          Eigen::Matrix<double, 2, 6> numeric_jacobian;
          for (int i = 0; i < 6; i++) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(i % 3) = eps;
            if (i <= 2) {
              q_temp = Qj;
              t_temp = Pj + delta;
            } else {
              Eigen::Matrix<double, 3, 1> half_theta = delta;
              half_theta /= static_cast<double>(2.0);
              Eigen::Quaternion<double> dq(1.0, half_theta.x(), half_theta.y(),
                                           half_theta.z());
              q_temp = (Qj * dq);
              t_temp = Pj;
            }
            Eigen::Vector3d pts_camera_i = obs_pt_i_ / inverse_depth_i;
            Eigen::Vector3d pts_world = Qi * pts_camera_i + Pi;
            Eigen::Vector3d pts_camera_j =
                q_temp.inverse() * (pts_world - t_temp);
            Eigen::Vector2d _error;
            double depth_j = pts_camera_j.z();
            _error = (pts_camera_j / depth_j).head<2>() - obs_pt_j_.head<2>();
            numeric_jacobian.col(i) = (_error - error) / eps;
          }
          std::cout << "numeric jacobi pose j:\n " << numeric_jacobian
                    << std::endl;
        }
      }
      if (jacobians[2]) {
        Eigen::Map<Eigen::Vector2d> jacobians_inverse_depth_i(jacobians[2]);
        jacobians_inverse_depth_i = reduce * Rj.transpose() * Ri * obs_pt_i_ *
                                    -1.0 / (inverse_depth_i * inverse_depth_i);
        {
          std::cout << "analysis jacobi inv depth :\n"
                    << jacobians_inverse_depth_i << std::endl;
          const double eps = 1e-6;
          double inverse_depth_i_eps = inverse_depth_i + eps;
          Eigen::Matrix<double, 2, 1> numeric_jacobian;

          Eigen::Vector3d pts_camera_i = obs_pt_i_ / inverse_depth_i_eps;
          Eigen::Vector3d pts_world = Qi * pts_camera_i + Pi;
          Eigen::Vector3d pts_camera_j = Qj.inverse() * (pts_world - Pj);

          Eigen::Vector2d _error;
          double depth_j = pts_camera_j.z();
          _error = (pts_camera_j / depth_j).head<2>() - obs_pt_j_.head<2>();
          numeric_jacobian = (_error - error) / eps;
          std::cout << "numeric jacobi inv depth:\n " << numeric_jacobian
                    << std::endl;
        }
      }
    }

    return true;
  }

 private:
  Eigen::Vector3d obs_pt_i_, obs_pt_j_;
};

struct CameraFrame
{
  CameraFrame(const Eigen::Matrix3d& R, Eigen::Vector3d t)
      : Rwc(R), Qwc(R), twc(std::move(t)){};
  Eigen::Matrix3d Rwc;
  Eigen::Quaterniond Qwc;
  Eigen::Vector3d twc;

  // point id, observed in current pose
  std::unordered_map<int, Eigen::Vector3d> feature_ids_;
};

void GetSimDataInWordFrame(std::vector<CameraFrame>& camera_poses,
                           std::vector<Eigen::Vector3d>& points,
                           const int feature_num, const int pose_num,
                           const double radius) {
  for (int n = 0; n < pose_num; n++) {
    // equivalent to a quarter-quadrant rotation, no matter how many positions
    double theta = n * 2 * M_PI / (pose_num * 4);
    Eigen::Matrix3d R =
        Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::Vector3d t = Eigen::Vector3d(
        radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
    camera_poses.emplace_back(R, t);
  }

  std::default_random_engine generator;
  std::normal_distribution<double> noise_pdf(0.0, 1. / 1000.);
  for (int j = 0; j < feature_num; j++) {
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(4., 8.);

    Eigen::Vector3d pw(xy_rand(generator), xy_rand(generator),
                       z_rand(generator));
    points.push_back(pw);

    // observe in every frame
    for (int i = 0; i < pose_num; i++) {
      Eigen::Vector3d pc =
          camera_poses[i].Rwc.transpose() * (pw - camera_poses[i].twc);
      pc = pc / pc.z();
      pc[0] += noise_pdf(generator);
      pc[1] += noise_pdf(generator);
      camera_poses[i].feature_ids_.insert(std::make_pair(j, pc));
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging("BA");
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  int pose_num = FLAGS_pose_number;
  int feature_num_per_frame = FLAGS_feature_num_per_frame;
  double radius = FLAGS_radius;

  std::vector<CameraFrame> camera_poses;
  std::vector<Eigen::Vector3d> points;

  GetSimDataInWordFrame(camera_poses, points, feature_num_per_frame, pose_num,
                        radius);

  int pose_id = 0, inv_depth_id = 0;
  std::unordered_map<int, tceres::slam::PoseParameterBlock> poses_params;
  std::unordered_map<int, tceres::slam::InvDepthParametersBlock>
      inv_depths_params;

  tceres::Problem problem;
  for (const auto& pose : camera_poses) {
    // TODO
    tceres::LocalParameterization* local_parameterization =
        new tceres::slam::PoseLocalParameterization();
    // tceres::LocalParameterization* local_parameterization =
    //     new tceres::ProductParameterization(
    //         new
    //         tceres::EigenQuaternionParameterization,new tceres::IdentityParameterization(3)
    //         );

    poses_params.emplace(pose_id++,
                         tceres::slam::PoseParameterBlock(pose.Qwc, pose.twc));

    problem.AddParameterBlock(poses_params.at(pose_id - 1).values(),
                              tceres::slam::PoseParameterBlock::ndim_,
                              local_parameterization);
  }
  problem.SetParameterBlockConstant(poses_params.at(0).values());

  std::default_random_engine generator;
  std::normal_distribution<double> noise_pdf(0, 2.);
  std::vector<double> noise_inv_depth;
  for (auto i = 0; i < points.size(); i++) {
    // suppose that all feature start frame is first frame
    Eigen::Vector3d Pw = points[i];
    // get the point observed in first frame
    Eigen::Vector3d Pc =
        camera_poses[0].Rwc.transpose() * (Pw - camera_poses[0].twc);
    double noise_depth = Pc.z() + noise_pdf(generator);
    noise_inv_depth.push_back(noise_depth);

    inv_depths_params.emplace(
        inv_depth_id++, tceres::slam::InvDepthParametersBlock(noise_depth));

    problem.AddParameterBlock(inv_depths_params.at(inv_depth_id - 1).values(),
                              tceres::slam::InvDepthParametersBlock::ndim_);
    // projection error corresponding to each feature,frame0 is the start frame
    for (auto j = 1; j < camera_poses.size(); j++) {
      Eigen::Vector3d pt_i = camera_poses[0].feature_ids_.find(i)->second;
      Eigen::Vector3d pt_j = camera_poses[j].feature_ids_.find(i)->second;
      tceres::CostFunction* cost_function =
          new BundleAdjustmentCostFunction(pt_i, pt_j);
      problem.AddResidualBlock(cost_function, nullptr,
                               poses_params.at(0).values(),
                               poses_params.at(j).values(),
                               inv_depths_params.at(inv_depth_id - 1).values());
    }
  }

  tceres::Solver::Summary summary;
  tceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = tceres::DENSE_SCHUR;
  options.minimizer_type = tceres::TRUST_REGION;
  options.max_num_iterations = 100;
  // options.check_gradients = true;
  options.gradient_check_relative_precision = 1e-6;
  tceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport();
  std::cout << "\nCompare MonoBA results after opt..." << std::endl;
  std::cout.precision(4);
  for (int k = 0; k < points.size(); k += 1) {
    std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z()
              << " ,noise " << 1. / noise_inv_depth[k] << " ,opt "
              << inv_depths_params.at(k).getInvDepth() << std::endl;
  }
  std::cout << "------------ pose translation ----------------" << std::endl;
  for (int i = 0; i < poses_params.size(); ++i) {
    std::cout << "translation after opt: " << i << " :"
              << poses_params.at(i).getTranslation().transpose()
              << " || gt: " << camera_poses[i].twc.transpose() << std::endl;
  }
  return 0;
}