#ifndef VIO_HELLO_WORLD_CONSTANT_MOTION_MODEL_HPP
#define VIO_HELLO_WORLD_CONSTANT_MOTION_MODEL_HPP

#include <glog/logging.h>

#include <Eigen/Core>

#include "sophus/se3.hpp"

namespace viohw {

class ConstantMotionModel
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void applyMotionModel(Sophus::SE3d &Twc, double time) {
    if (prev_time_ > 0) {
      // Provided Twc and prevTwc should be equal here
      // as prevTwc is updated right after pose computation
      if (!(Twc * prevTwc_.inverse()).log().isZero(1.e-5)) {
        // Might happen in case of LC!
        // So update prevPose to stay consistent
        prevTwc_ = Twc;
      }

      double dt = (time - prev_time_);
      Twc = Twc * Sophus::SE3d::exp(log_relT_ * dt);
    }
  }

  void updateMotionModel(const Sophus::SE3d &Twc, double time) {
    if (prev_time_ < 0.) {
      prev_time_ = time;
      prevTwc_ = Twc;
    } else {
      double dt = time - prev_time_;

      prev_time_ = time;

      if (dt < 0.) {
        LOG(FATAL) << "\nGot image older than previous image! LEAVING!\n";
      }

      Sophus::SE3d Tprevcur = prevTwc_.inverse() * Twc;
      log_relT_ = Tprevcur.log() / dt;
      prevTwc_ = Twc;
    }
  }

  void reset() {
    prev_time_ = -1.;
    log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
  }

  double prev_time_ = -1.;

  Sophus::SE3d prevTwc_;
  Eigen::Matrix<double, 6, 1> log_relT_ = Eigen::Matrix<double, 6, 1>::Zero();
};
}  // namespace viohw

#endif  // VIO_HELLO_WORLD_CONSTANT_MOTION_MODEL_HPP
