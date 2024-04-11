#pragma once
#include <glog/logging.h>

#include <queue>
#include <thread>

#include "backend/sensor_fusion/imu/imu_types.h"

namespace backend {
namespace IMU {

class IMUDataBase
{
public:
  enum IMUMeasureStatus
  {
    IMU_QUEEN_EMPTY,
    PREV_IMAGE_TIME_LARGER_IMU_END_TIME,
    CURR_IMAGE_TIME_LESS_IMU_FRONT_TIME,
    GET_LOST_IMU_DATA,
    SUCCESS_GET_IMU_DATA,
  };

  IMUDataBase() = default;

  IMUDataBase( int imu_FEQ, int cam_FEQ );

  ~IMUDataBase() = default;

  void InsertMeasure( const IMU::Point &msg );

  size_t GetDatabaseSize() const { return queen_imu_meas_.size(); }

  void EraseOlderIMUMeasure( double time );

  bool PreIntegrateIMU( std::vector<backend::IMU::Point> &imus, double last_image_time,
                        double curr_image_time,
                        const IMU::PreintegratedPtr& pre_integrated_from_last_frame,
                        const IMU::PreintegratedPtr& pre_integrated_from_last_kf );

  IMUMeasureStatus GetIntervalIMUMeasurement( double start, double end,
                                              std::vector<IMU::Point> &data );

private:
  int IMU_FEQ, CAM_FEQ;
  int min_interval_imu_size_ = 0;
  std::mutex imu_meas_mutex_;
  std::queue<IMU::Point> queen_imu_meas_;
};

typedef std::shared_ptr<IMUDataBase> IMUDataBasePtr;
typedef std::shared_ptr<const IMUDataBase> IMUDataBaseConstPtr;

inline void IMUDataBase::EraseOlderIMUMeasure( double time ) {
  double oldest_imu_time = queen_imu_meas_.front().t;
  if ( time <= oldest_imu_time ) {
    return;
  }
  while ( time < queen_imu_meas_.front().t ) {
    queen_imu_meas_.pop();
  }
}

inline IMUDataBase::IMUDataBase( const int imu_FEQ, const int cam_FEQ ) {
  CAM_FEQ = cam_FEQ;
  IMU_FEQ = imu_FEQ;
  min_interval_imu_size_ = IMU_FEQ / CAM_FEQ;
}

inline void IMUDataBase::InsertMeasure( const IMU::Point &msg ) {
  std::unique_lock<std::mutex> lck( imu_meas_mutex_ );
  queen_imu_meas_.push( msg );
}

inline IMUDataBase::IMUMeasureStatus IMUDataBase::GetIntervalIMUMeasurement(
    double start, double end, std::vector<IMU::Point> &data ) {
  data.clear();

  std::unique_lock<std::mutex> lck( imu_meas_mutex_ );
  if ( queen_imu_meas_.empty() ) {
    return IMU_QUEEN_EMPTY;
  }

  double queen_start_time = queen_imu_meas_.front().t;
  double queen_end_time = queen_imu_meas_.back().t;

  if ( start > queen_end_time ) {
    return PREV_IMAGE_TIME_LARGER_IMU_END_TIME;
  }

  if ( end < queen_start_time ) {
    return CURR_IMAGE_TIME_LESS_IMU_FRONT_TIME;
  }

  // pop IMU data less than the starting image time
  while ( !queen_imu_meas_.empty() && queen_imu_meas_.front().t <= start ) {
    queen_imu_meas_.pop();
  }

  while ( !queen_imu_meas_.empty() && queen_imu_meas_.front().t <= end ) {
    data.push_back( queen_imu_meas_.front() );
    queen_imu_meas_.pop();
  }

  if ( data.size() < min_interval_imu_size_ ) {
    return GET_LOST_IMU_DATA;
  }

  return SUCCESS_GET_IMU_DATA;
}

inline bool IMUDataBase::PreIntegrateIMU( std::vector<backend::IMU::Point> &imus,
                                          double last_image_time, double curr_image_time,
                                          const IMU::PreintegratedPtr& pre_integrated_from_last_frame,
                                          const IMU::PreintegratedPtr& pre_integrated_from_last_kf ) {
  if ( imus.empty() ) {
    return false;
  }

  IMU::Preintegrated pre_integrate;

  int n = imus.size();
  for ( int i = 0; i < n; i++ ) {
    double tstep;
    Eigen::Vector3d acc, angVel;
    if ( ( i == 0 ) && ( i < ( n - 1 ) ) ) {
      double tab = imus[i + 1].t - imus[i].t;
      double tini = imus[i].t - last_image_time;
      acc = ( imus[i].a + imus[i + 1].a - ( imus[i + 1].a - imus[i].a ) * ( tini / tab ) ) * 0.5f;
      angVel =
          ( imus[i].w + imus[i + 1].w - ( imus[i + 1].w - imus[i].w ) * ( tini / tab ) ) * 0.5f;
      tstep = imus[i + 1].t - last_image_time;
    } else if ( i < ( n - 1 ) ) {
      acc = ( imus[i].a + imus[i + 1].a ) * 0.5f;
      angVel = ( imus[i].w + imus[i + 1].w ) * 0.5f;
      tstep = imus[i + 1].t - imus[i].t;
    } else if ( ( i > 0 ) && ( i == ( n - 1 ) ) ) {
      double tab = imus[i + 1].t - imus[i].t;
      double tend = imus[i + 1].t - curr_image_time;
      acc = ( imus[i].a + imus[i + 1].a - ( imus[i + 1].a - imus[i].a ) * ( tend / tab ) ) * 0.5f;
      angVel =
          ( imus[i].w + imus[i + 1].w - ( imus[i + 1].w - imus[i].w ) * ( tend / tab ) ) * 0.5f;
      tstep = curr_image_time - imus[i].t;
    } else if ( ( i == 0 ) && ( i == ( n - 1 ) ) ) {
      acc = imus[i].a;
      angVel = imus[i].w;
      tstep = curr_image_time - last_image_time;
    }
    pre_integrated_from_last_frame->IntegrateNewMeasurement( acc.cast<float>(),
                                                             angVel.cast<float>(), tstep );
    pre_integrated_from_last_kf->IntegrateNewMeasurement( acc.cast<float>(), angVel.cast<float>(),
                                                          tstep );
  }
}

}  // namespace IMU

typedef IMU::IMUDataBase::IMUMeasureStatus IMUMeasureStatus;

#define CASESTR( x ) \
  case x:            \
    return #x

inline const char *IMUMeasureStatusToString( IMUMeasureStatus type ) {
  switch ( type ) {
    CASESTR( IMUMeasureStatus::IMU_QUEEN_EMPTY );
    CASESTR( IMUMeasureStatus::PREV_IMAGE_TIME_LARGER_IMU_END_TIME );
    CASESTR( IMUMeasureStatus::CURR_IMAGE_TIME_LESS_IMU_FRONT_TIME );
    CASESTR( IMUMeasureStatus::GET_LOST_IMU_DATA );
    CASESTR( IMUMeasureStatus::SUCCESS_GET_IMU_DATA );
  }
}

#undef CASESTR

}  // namespace backend