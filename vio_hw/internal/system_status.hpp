#pragma once
#include <atomic>
#include <mutex>

namespace viohw {

enum TrackingStatus
{
  TrackLost,
  TrackBad,
  TrackGood
};

enum InitStatus
{
  NotInit,
  InitSuccess,
  InitFailed
};

class SystemState
{
public:
  SystemState() = default;
  ~SystemState() = default;

  TrackingStatus getTrackerStatus() const {
    std::lock_guard<std::mutex> lck( tracker_status_mutex_ );
    return tracking_status_;
  }

  void setTrackerStatus( const TrackingStatus &status ) {
    std::lock_guard<std::mutex> lck( tracker_status_mutex_ );
    tracking_status_ = status;
  }

  InitStatus getInitStatus() const {
    std::lock_guard<std::mutex> lck( init_status_mutex_ );
    return init_status_;
  }

  void setInitStatus( const InitStatus &status ) {
    std::lock_guard<std::mutex> lck( init_status_mutex_ );
    init_status_ = status;
  }

  std::atomic<bool> is_local_ba_{ false };
  std::atomic<bool> is_request_reset_{ false };

private:
  mutable std::mutex tracker_status_mutex_;
  mutable std::mutex init_status_mutex_;
  TrackingStatus tracking_status_ = TrackGood;
  InitStatus init_status_ = NotInit;
};

typedef std::shared_ptr<SystemState> SystemStatePtr;
typedef std::shared_ptr<const SystemState> SystemStateConstPtr;

}  // namespace viohw
