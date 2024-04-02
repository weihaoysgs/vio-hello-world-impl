#ifndef VIO_HELLO_WORLD_LOOP_CLOSER_HPP
#define VIO_HELLO_WORLD_LOOP_CLOSER_HPP

#include <chrono>
#include <memory>
#include <thread>

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"
#include "vio_hw/internal/map_manager.hpp"
#include "vio_hw/internal/setting.hpp"

namespace viohw {

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

class LoopCloser
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  LoopCloser() = default;

  ~LoopCloser() = default;

  // construction function
  LoopCloser( SettingPtr param, MapManagerPtr map_manager );

  // loop closer thread
  void run();

  // get new keyframe for loop closer detect
  bool GetNewKeyFrame();

  // add new keyframe to loop closer thread
  void AddNewKeyFrame( const FramePtr &pkf, const cv::Mat &im );

  // compute desc of new kf image
  void ComputeDesc( std::vector<cv::KeyPoint> &cv_kps, cv::Mat &cv_descs );

  // detect loop closer for new kf
  std::pair<int, float> DetectLoop( cv::Mat &desc );

  // process loop kf for pose graph optimization
  void ProcessLoopCandidate( int kf_loop_id );

  void KNNMatching( const Frame &newkf, const Frame &lckf,
                    std::vector<std::pair<int, int>> &vkplmids );

  // desc convert tool, ref ORB-SLAM3
  static std::vector<cv::Mat> ConvertToDescriptorVector( const cv::Mat &descriptors );

private:
  cv::Mat new_kf_img_;

  FramePtr new_kf_;
  SettingPtr param_;
  MapManagerPtr map_manager_;

  bool is_new_kf_available_ = false;

  std::queue<std::pair<FramePtr, cv::Mat>> kfs_queen_;
  std::mutex kf_queen_mutex_;

  cv::Ptr<cv::FeatureDetector> fast_detect_;
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief_cal_;
  std::unique_ptr<ORBVocabulary> ORBVocabulary_;
  std::map<int, DBoW2::BowVector> map_kf_bow_vec_;
  double loop_threshold_ = 0.6;
  bool use_loop_;
};

typedef std::shared_ptr<LoopCloser> LoopCloserPtr;
typedef std::shared_ptr<const LoopCloser> LoopCloserConstPtr;

}  // namespace viohw
#endif  // VIO_HELLO_WORLD_LOOP_CLOSER_HPP
