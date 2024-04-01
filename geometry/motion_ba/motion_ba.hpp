#ifndef VIO_HELLO_WORLD_MOTION_BA_HPP
#define VIO_HELLO_WORLD_MOTION_BA_HPP

#include "backend/cost_function/reprojection_se3_factor.hpp"
#include "backend/parameter_block/point_parameter_block.hpp"
#include "backend/parameter_block/se3_parameter_block.hpp"
#include "backend/parameterization/se3left_parametrization.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "tceres/loss_function.h"
#include "tceres/problem.h"
#include "tceres/solver.h"

namespace geometry {

bool tceresMotionOnlyBA(
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const std::vector<int> &vscales, Sophus::SE3d &Twc, const int nmaxiter, const float chi2th,
    const bool buse_robust, const bool bapply_l2_after_robust, const float fx, const float fy,
    const float cx, const float cy, std::vector<int> &voutliersidx);

bool tceresMotionOnlyBA(
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &vunkps,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const std::vector<int> &vscales, Sophus::SE3d &Twc, const int nmaxiter, const float chi2th,
    const bool buse_robust, const bool bapply_l2_after_robust, const Eigen::Matrix3d &K,
    std::vector<int> &voutliersidx);

bool opencvP3PRansac(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &vwpts,
    const int nmaxiter, const float errth, const float fx, const float fy, const bool boptimize,
    Sophus::SE3d &Twc, std::vector<int> &voutliersidx);
}  // namespace geometry

#endif  // VIO_HELLO_WORLD_MOTION_BA_HPP
