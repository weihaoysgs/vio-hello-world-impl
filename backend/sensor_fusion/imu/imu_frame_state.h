#pragma once
#include <mutex>

#include "backend/sensor_fusion/imu/imu_types.h"

namespace backend {

namespace IMU {

class IMUState
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setIMUMeasureAvailable( bool flag ) {
    std::lock_guard<std::mutex> lck( imu_state_mutex_ );
    imu_measure_available_ = flag;
  }

  void setImuStateVelocity( const Eigen::Vector3d &vel ) {
    std::lock_guard<std::mutex> lck( imu_state_mutex_ );
    Vel_ = vel;
  }

  Eigen::Vector3d getImuStateVelocity() const {
    std::lock_guard<std::mutex> lck( imu_state_mutex_ );
    return Vel_;
  };

  Sophus::SE3d getTwb() const {
    std::lock_guard<std::mutex> lck( imu_state_mutex_ );
    return Twb_;
  }

  void setTwb( const Sophus::SE3d &Twb ) {
    std::lock_guard<std::mutex> lck( imu_state_mutex_ );
    Twb_ = Tbw_;
    Tbw_ = Twb_.inverse();
  }

  bool getImuMeasureAvailable() const {
    std::lock_guard<std::mutex> lck( imu_state_mutex_ );
    return imu_measure_available_;
  }

  void resetPreIntegrationFromLastFrame( const backend::IMU::Bias &bias,
                                         const backend::IMU::Calib &calib ) {
    preintegrated_from_last_frame_.reset( new backend::IMU::Preintegrated( bias, calib ) );
  }

  void resetPreIntegrationFromLastKF( const backend::IMU::Bias &bias,
                                      const backend::IMU::Calib &calib ) {
    preintegrated_from_last_kf_.reset( new backend::IMU::Preintegrated( bias, calib ) );
  }

  // imu pre integrate measure
  backend::IMU::PreintegratedPtr preintegrated_from_last_frame_;
  backend::IMU::PreintegratedPtr preintegrated_from_last_kf_;

private:
  mutable std::mutex imu_state_mutex_;

  bool imu_measure_available_ = false;
  Sophus::SE3d Twb_, Tbw_;
  Eigen::Vector3d Vel_;
};

typedef std::shared_ptr<IMUState> IMUStatePtr;
typedef std::shared_ptr<const IMUState> IMUStateConstPtr;
}  // namespace IMU

}  // namespace backend