// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include "feat/harris/harris.h"

#include "common/draw_utils.h"
#include "common/wall_time.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

DEFINE_string(image_path, "../feat/data/euroc01.png", "image file path");
DEFINE_int32(max_kps_num, 500, "max detector features num");
DEFINE_double(kps_min_distance, 30, "min kp distance");
DEFINE_double(kps_quality_level, 0.01, "qualityLevel of kp");

cv::Mat openCVMethod(const cv::Mat& image, int max_kps_num,
                     double kps_min_distance, double kps_quality_level) {
  std::vector<cv::Point2f> kps;

  com::EvaluateAndCall(
      [&]() {
        kps.clear();
        cv::goodFeaturesToTrack(image, kps, max_kps_num, kps_quality_level,
                                kps_min_distance, cv::Mat());
      },
      "cv::goodFeaturesToTrack", 20);

  return com::DrawKeyPoints(image, kps);
}

cv::Mat harrisDetector(const cv::Mat& image, int max_kps_num,
                       double kps_min_distance, double kps_quality_level) {
  std::vector<cv::Point2f> kps;

  com::EvaluateAndCall(
      [&]() {
        kps.clear();
        feat::goodFeaturesToTrack(image, kps, max_kps_num, kps_quality_level,
                                  kps_min_distance, cv::Mat());
      },
      "feat::goodFeaturesToTrack", 20);

  return com::DrawKeyPoints(image, kps);
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging("harris_test");
  FLAGS_stderrthreshold = google::INFO;
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat gray_image = cv::imread(FLAGS_image_path, cv::IMREAD_GRAYSCALE);

  int max_kps_num = FLAGS_max_kps_num;
  double kps_min_distance = FLAGS_kps_min_distance;
  double kps_quality_level = FLAGS_kps_quality_level;

  LOG_ASSERT(gray_image.empty() == false)
      << " Image is empty, please check you image path.";

  cv::Mat opencv_harris = openCVMethod(gray_image, max_kps_num,
                                       kps_min_distance, kps_quality_level);
  cv::Mat self_harris = harrisDetector(gray_image, max_kps_num,
                                       kps_min_distance, kps_quality_level);

  cv::imshow("opencv_harris", opencv_harris);
  cv::imshow("self_harris", self_harris);
  cv::waitKey(0);

  return 0;
}