#ifndef VIO_HELLO_WORLD_WALL_TIME_HPP
#define VIO_HELLO_WORLD_WALL_TIME_HPP
#include <stdio.h>
#include <sys/time.h>

#include "glog/logging.h"

namespace com {

template <int = 0>
double WallTimeInSeconds() {
  timeval time_val;
  gettimeofday(&time_val, NULL);
  return (time_val.tv_sec + time_val.tv_usec * 1e-6);
}

template <typename FuncT>
void EvaluateAndCall(FuncT&& func, const std::string& func_name = "",
                     int times = 10) {
  double total_time = 0;
  for (int i = 0; i < times; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    func();
    auto t2 = std::chrono::high_resolution_clock::now();
    // clang-format off
    total_time += std::chrono::duration_cast<
                  std::chrono::duration<double>>(t2 - t1).count() * 1000;
    // clang-format on
  }

  LOG(INFO) << "Function " << func_name << " call " << times
            << " times, average cost per evaluate is " << total_time / times
            << " ms.";
}

}  // namespace com

#endif  // VIO_HELLO_WORLD_WALL_TIME_HPP
