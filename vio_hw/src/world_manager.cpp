#include "vio_hw/vio/world_manager.hpp"

namespace viohw {
WorldManager::WorldManager(std::shared_ptr<Setting>& setting)
    : params_(setting) {}


void WorldManager::run() {
  while (true) {
    std::printf("World Manager running\n");
    std::chrono::milliseconds dura(1000);
    std::this_thread::sleep_for(dura);
  }
}
}  // namespace viohw