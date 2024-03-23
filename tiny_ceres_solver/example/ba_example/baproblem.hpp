#ifndef BAPROBLEM_HPP_
#define BAPROBLEM_HPP_

#include "parametersse3.hpp"
#include "tceres/problem.h"

class Sample
{
 public:
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr) {
  return lowerBndr +
         ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma) {
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to) {
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform() { return uniform_rand(0., 1.); }

double Sample::gaussian(double sigma) { return gauss_rand(0., sigma); }
/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template <int PoseBlockSize>
class BAProblem
{
 public:
  BAProblem(int pose_num_, int point_num_, double pix_noise_,
            bool useOrdering = false);
  void solve(tceres::Solver::Options& opt, tceres::Solver::Summary* sum);
  tceres::Problem problem;
  PosePointParametersBlock<PoseBlockSize> states;
  PosePointParametersBlock<PoseBlockSize> true_states;
  std::vector<Eigen::Vector3d> noise_points;
  struct EigenPose
  {
    Eigen::Quaterniond Q;
    Eigen::Vector3d t;
  };
  std::vector<EigenPose> before_opt_pose;
};

template <int PoseBlockSize>
BAProblem<PoseBlockSize>::BAProblem(int pose_num_, int point_num_,
                                    double pix_noise_, bool useOrdering) {
  int pose_num = pose_num_;
  int point_num = point_num_;
  double PIXEL_NOISE = pix_noise_;

  states.create(pose_num, point_num);
  true_states.create(pose_num, point_num);

  for (int i = 0; i < point_num; ++i) {
    Eigen::Map<Vector3d> true_pt(true_states.point(i));
    true_pt = Vector3d((Sample::uniform() - 0.5) * 3, Sample::uniform() - 0.5,
                       Sample::uniform() + 3);
  }

  double focal_length = 1000.;
  double cx = 320.;
  double cy = 240.;
  CameraParameters cam(focal_length, cx, cy);

  for (int i = 0; i < pose_num; ++i) {
    Vector3d trans(i * 0.04 - 1., 0, 0);

    Eigen::Quaterniond q;
    q.setIdentity();
    true_states.setPose(i, q, trans);
    states.setPose(i, q, trans);

    EigenPose eigen_pose;
    eigen_pose.Q = q;
    eigen_pose.t = trans;
    before_opt_pose.emplace_back(eigen_pose);

    problem.AddParameterBlock(states.pose(i), PoseBlockSize,
                              new PoseSE3Parameterization<PoseBlockSize>());

    if (i < 2) {
      problem.SetParameterBlockConstant(states.pose(i));
    }
  }

  for (int i = 0; i < point_num; ++i) {
    Eigen::Map<Vector3d> true_point_i(true_states.point(i));
    Eigen::Map<Vector3d> noise_point_i(states.point(i));
    noise_point_i =
        true_point_i +
        Vector3d(Sample::gaussian(1), Sample::gaussian(1), Sample::gaussian(1));

    noise_points.emplace_back(noise_point_i);

    Vector2d z;
    SE3 true_pose_se3;

    int num_obs = 0;
    for (int j = 0; j < pose_num; ++j) {
      true_states.getPose(j, true_pose_se3.rotation(),
                          true_pose_se3.translation());
      Vector3d point_cam = true_pose_se3.map(true_point_i);
      z = cam.cam_map(point_cam);
      if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
        ++num_obs;
      }
    }
    if (num_obs >= 2) {
      problem.AddParameterBlock(states.point(i), 3);
      // if(useOrdering)
      //     ordering->AddElementToGroup(states.point(i), 0);

      for (int j = 0; j < pose_num; ++j) {
        true_states.getPose(j, true_pose_se3.rotation(),
                            true_pose_se3.translation());
        Vector3d point_cam = true_pose_se3.map(true_point_i);
        z = cam.cam_map(point_cam);

        if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
          z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                        Sample::gaussian(PIXEL_NOISE));

          tceres::CostFunction* costFunc =
              new ReprojectionErrorSE3XYZ<PoseBlockSize>(focal_length, cx, cy,
                                                         z[0], z[1]);
          problem.AddResidualBlock(costFunc, NULL, states.pose(j),
                                   states.point(i));
        }
      }
    }
  }
}

template <int PoseBlockSize>
void BAProblem<PoseBlockSize>::solve(tceres::Solver::Options& opt,
                                     tceres::Solver::Summary* sum) {
  tceres::Solve(opt, &problem, sum);
}

#endif
