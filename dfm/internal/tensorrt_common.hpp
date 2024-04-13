#ifndef TENSORRT_COMMON_HPP
#define TENSORRT_COMMON_HPP

#include <chrono>

#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/common.h"

namespace samplesCommon {
template <typename T>
using trtUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
}  // namespace samplesCommon

namespace dfm {

namespace trtCommon = samplesCommon;
namespace trtLogger = sample;
namespace trtType = sample;

class TimeLog
{
public:
  TimeLog() { start(); }
  void start() { start_ = std::chrono::steady_clock::now(); }

  double end() {
    end_ = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end_ - start_ );
    return duration.count();
  }

  void logTime( std::string log ) {
    double cost_time = end();
    std::cout << "[Time:] " << log << " cost time " << cost_time << " ms\n";
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start_, end_;
};

}  // namespace dfm

#endif  // TENSORRT_COMMON_HPP