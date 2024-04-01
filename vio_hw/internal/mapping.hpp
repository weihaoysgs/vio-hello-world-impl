#ifndef VIO_HELLO_WORLD_MAPPING_HPP
#define VIO_HELLO_WORLD_MAPPING_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <mutex>
#include <thread>

#include "sophus/se3.hpp"
#include "vio_hw/internal/frame.hpp"
#include "vio_hw/internal/keyframe.hpp"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"
#include "geometry/triangulate/triangulate_cv.hpp"

namespace viohw {

class Mapping
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  // default construction
  Mapping() = default;

  // default deconstruction
  ~Mapping() = default;

  // Mapping construction
  Mapping(SettingPtr param, MapManagerPtr map_manager, FramePtr frame);

  // Mapping thread
  void run();

  // get new keyframe, return true meaning successful
  bool GetNewKf(Keyframe &kf);

  // add new kf to kf queen wait for process
  void AddNewKf(const Keyframe &kf);

  void TriangulateTemporal(Frame &frame);

  void TriangulateStereo(Frame &frame);

  Eigen::Vector3d ComputeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl,
                                       const Eigen::Vector3d &bvr);

 private:
  bool is_new_kf_available_ = false;
  bool stereo_mode_;
  bool use_clahe_;
  cv::Ptr<cv::CLAHE> clahe_;

  FramePtr current_frame_;
  SettingPtr params_;
  MapManagerPtr map_manager_;

  std::queue<Keyframe> kfs_queen_;

  std::mutex kf_queen_mutex_;
};

typedef std::shared_ptr<Mapping> MappingPtr;
typedef std::shared_ptr<const Mapping> MappingConstPtr;

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_MAPPING_HPP
