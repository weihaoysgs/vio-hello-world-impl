#include "vio_hw/internal/mapping.hpp"

namespace viohw {

Mapping::Mapping(viohw::SettingPtr param, viohw::MapManagerPtr map_manager, viohw::FramePtr frame)
    : params_(param), map_manager_(map_manager), current_frame_(frame) {
  std::thread mapper_thread(&Mapping::run, this);
  mapper_thread.detach();
}

void Mapping::run() {
  LOG(INFO) << "Mapper is ready to process Keyframes!";
  Keyframe kf;
  while (true) {
    if (getNewKf(kf)) {
    } else {
      std::chrono::microseconds dura(100);
      std::this_thread::sleep_for(dura);
    }
  }
}

void Mapping::TriangulateTemporal(Frame& frame) {}

void Mapping::TriangulateStereo(Frame& frame) {}

Eigen::Vector3d Mapping::ComputeTriangulation(const Sophus::SE3d& T, const Eigen::Vector3d& bvl,
                                              const Eigen::Vector3d& bvr) {
  return Eigen::Vector3d();
}

bool Mapping::getNewKf(Keyframe& kf) {
  std::lock_guard<std::mutex> lock(kf_queen_mutex_);

  // Check if new KF is available
  if (kfs_queen_.empty()) {
    is_new_kf_available_ = false;
    return false;
  }

  // Get new KF and signal BA to stop if
  // it is still processing the previous KF
  kf = kfs_queen_.front();
  kfs_queen_.pop();

  // Setting is_new_kf_available_ to true will limit
  // the processing of the KF to triangulation and postpone
  // other costly tasks to next KF as we are running late!
  if (kfs_queen_.empty()) {
    is_new_kf_available_ = false;
  } else {
    is_new_kf_available_ = true;
  }

  return true;
}

void Mapping::addNewKf(const Keyframe& kf) {
  std::lock_guard<std::mutex> lock(kf_queen_mutex_);

  kfs_queen_.push(kf);

  is_new_kf_available_ = true;
}

}  // namespace viohw
