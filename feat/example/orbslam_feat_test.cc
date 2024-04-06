// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include "common/draw_utils.h"
#include "common/wall_time.h"
#include "feat/orb/orb_feature.h"
#include "feat/orb_slam/orbextractor.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string( image_path, "../feat/data/euroc01.png", "image file path" );
DEFINE_int32( max_kps, 500, "max kps" );

cv::Mat ORBImpl( const cv::Mat& image ) {
  auto orb = feat::ORB::create( FLAGS_max_kps );
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Mat roi( image.size(), CV_8UC1, cv::Scalar( 255 ) );

  com::EvaluateAndCall( [&]() { orb->detectAndCompute( image, roi, keypoints, descriptors ); },
                        "ORBImpl", 20 );

  return com::DrawKeyPoints( image, keypoints, "" );
}

cv::Mat OpenCVORB( const cv::Mat& image ) {
  auto orb = cv::ORB::create( FLAGS_max_kps );
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Mat roi( image.size(), CV_8UC1, cv::Scalar( 255 ) );

  com::EvaluateAndCall( [&]() { orb->detectAndCompute( image, roi, keypoints, descriptors ); },
                        "OpenCVORB", 20 );

  return com::DrawKeyPoints( image, keypoints, "" );
}

cv::Mat ORB_SLAM_Feat( const cv::Mat& image ) {
  std::shared_ptr<feat::ORBextractor> orb_extractor_ = nullptr;
  orb_extractor_.reset( new feat::ORBextractor( FLAGS_max_kps, 1.2, 8, 20, 7 ) );
  std::vector<cv::KeyPoint> kps;
  cv::Mat mask( image.size(), CV_8UC1, cv::Scalar::all( 255 ) );
  com::EvaluateAndCall( [&]() { orb_extractor_->Detect( image, mask, kps ); }, "ORB-SLAM Feat",
                        20 );
  return com::DrawKeyPoints( image, kps, "" );
}

int main( int argc, char** argv ) {
  ::google::InitGoogleLogging( "test_orb_feature" );
  FLAGS_stderrthreshold = google::INFO;
  ::google::ParseCommandLineFlags( &argc, &argv, true );

  cv::Mat gray_image = cv::imread( FLAGS_image_path, cv::IMREAD_GRAYSCALE );
  LOG_ASSERT( gray_image.empty() == false ) << " Image is empty, please check you image path.";

  cv::Mat orb_impl_result = ORBImpl( gray_image );
  cv::Mat cv_impl_result = OpenCVORB( gray_image );
  cv::Mat orbslam_feat_result = ORB_SLAM_Feat( gray_image );

  cv::imshow( "ORB Impl", orb_impl_result );
  cv::imshow( "ORBSLAM Feat Impl", orbslam_feat_result );
  cv::imshow( "OpenCV Impl", cv_impl_result );
  cv::waitKey( 0 );
  return 0;
}
