// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

#include "common/draw_utils.h"
#include "common/wall_time.h"
#include "feat/harris/harris.h"
#include "feat/optical_flow/lkpyramid.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

DEFINE_string(image0_path, "../feat/data/optical_flow1.png", "image file path");
DEFINE_string(image1_path, "../feat/data/optical_flow2.png", "image file path");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging("harris_test");
  FLAGS_stderrthreshold = google::INFO;
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  cv::Mat gray_image0 = cv::imread(FLAGS_image0_path, cv::IMREAD_GRAYSCALE);
  cv::Mat gray_image1 = cv::imread(FLAGS_image1_path, cv::IMREAD_GRAYSCALE);
  LOG_ASSERT(gray_image0.empty() == false)
      << " Image0 is empty, please check you image path.";
  LOG_ASSERT(gray_image1.empty() == false)
      << " Image1 is empty, please check you image path.";
  std::vector<cv::Point2f> image0_kps, image1_kps;
  feat::goodFeaturesToTrack(gray_image0, image0_kps, 200, 0.01, 30, cv::Mat());
  image1_kps = image0_kps;

  std::vector<uchar> status;
  com::EvaluateAndCall(
      [&]() {
        feat::calcOpticalFlowPyrLK(
            gray_image0, gray_image1, image0_kps, image1_kps, status,
            cv::noArray(), cv::Size(20, 20), 4,
            cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                             30, 0.01f),
            0);
      },
      "feat::calcOpticalFlowPyrLK", 20);
  com::EvaluateAndCall(
      [&]() {
        cv::calcOpticalFlowPyrLK(
            gray_image0, gray_image1, image0_kps, image1_kps, status,
            cv::noArray(), cv::Size(20, 20), 4,
            cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                             30, 0.01f),
            0);
      },
      "cv::calcOpticalFlowPyrLK", 20);

  LOG(INFO) << "status.size() " << status.size() << "," << image0_kps.size()
            << "," << image1_kps.size() << std::endl;
  cv::Mat draw_tracker_image;
  cv::cvtColor(gray_image0, draw_tracker_image, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < status.size(); i++) {
    if (status[i]) {
      cv::arrowedLine(draw_tracker_image, image0_kps[i], image1_kps[i],
                      cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
  }
  cv::imshow("feat tracker result", draw_tracker_image);
  cv::waitKey(0);
  return 0;
}