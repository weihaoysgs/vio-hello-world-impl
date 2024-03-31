#include "vio_hw/internal//map_point.hpp"

namespace viohw {
MapPoint::MapPoint(const int lmid, const int kfid, const bool bobs)
    : lmid_(lmid), isobs_(bobs), kfid_(kfid), invdepth_(-1.) {
  set_kfids_.insert(kfid);
  is3d_ = false;
  ptxyz_.setZero();
}

MapPoint::MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const bool bobs)
    : lmid_(lmid), isobs_(bobs), kfid_(kfid), invdepth_(-1.) {
  set_kfids_.insert(kfid);

  map_kf_desc_.emplace(kfid, desc);
  map_desc_dist_.emplace(kfid, 0.);
  desc_ = map_kf_desc_.at(kfid);

  is3d_ = false;
  ptxyz_.setZero();
}
}  // namespace viohw