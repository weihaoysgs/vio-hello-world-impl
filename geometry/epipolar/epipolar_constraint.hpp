#ifndef VIO_HELLO_WORLD_EPIPOLAR_CONSTRAINT_HPP
#define VIO_HELLO_WORLD_EPIPOLAR_CONSTRAINT_HPP

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace geometry {
bool Opencv5ptEssentialMatrix(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs1,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &bvs2,
    const int nmaxiter, const float errth, const bool boptimize, const float fx, const float fy,
    Eigen::Matrix3d &Rwc, Eigen::Vector3d &twc, std::vector<int> &voutliersidx );

}
#endif  // VIO_HELLO_WORLD_EPIPOLAR_CONSTRAINT_HPP
