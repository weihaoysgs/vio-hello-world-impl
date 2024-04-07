#ifndef VIO_HELLO_WORLD_MAP_POINT_HPP
#define VIO_HELLO_WORLD_MAP_POINT_HPP
#include <Eigen/Core>
#include <mutex>
#include <opencv2/core.hpp>
#include <set>
#include <unordered_map>
namespace viohw {

class MapPoint
{
 public:
  enum MapPointColor
  {
    OBSERVED,
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MapPoint(const int lmid, const int kfid, const bool bobs = true);
  MapPoint(const int lmid, const int kfid, const cv::Mat &desc, const bool bobs = true);

  std::set<int> GetKfObsSet() const;

  void RemoveKfObs(const int kfid);

  Eigen::Vector3d GetPoint() const;

  void SetPoint(const Eigen::Vector3d &ptxyz, const double invdepth = -1.);

  void AddKfObs(const int kfid);

  void AddDesc(const int kfid, const cv::Mat &d);

  // For using MapPoint in ordered containers
  bool operator<(const MapPoint &mp) const { return lmid_ < mp.lmid_; }

  // MapPoint id
  int lmid_;

  // True if seen in current frame
  bool isobs_;

  // True if MP has been init
  bool is3d_;

  // Set of observed KF ids
  std::set<int> set_kfids_;

  // 3D position
  Eigen::Vector3d ptxyz_;

  // Anchored position
  int kfid_;
  double invdepth_;

  // Mean desc and list of descs
  cv::Mat desc_;
  std::unordered_map<int, cv::Mat> map_kf_desc_;
  std::unordered_map<int, float> map_desc_dist_;

  // For vizu
  cv::Scalar color_ = cv::Scalar(200);

  mutable std::mutex pt_mutex;
};
}  // namespace viohw

#endif  // VIO_HELLO_WORLD_MAP_POINT_HPP
