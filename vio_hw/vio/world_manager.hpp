//
// Created by weihao on 24-3-25.
//

#ifndef VIO_HELLO_WORLD_WORLD_MANAGER_HPP
#define VIO_HELLO_WORLD_WORLD_MANAGER_HPP

#include <chrono>
#include <thread>

#include "vio_hw/vio/setting.hpp"

namespace viohw {
class WorldManager
{
 public:
  explicit WorldManager(std::shared_ptr<Setting>& setting);
  void run();
  const std::shared_ptr<Setting> getParams() const { return params_; }

 private:
  const std::shared_ptr<Setting> params_;
};
}  // namespace viohw
#endif  // VIO_HELLO_WORLD_WORLD_MANAGER_HPP
