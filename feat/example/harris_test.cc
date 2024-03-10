#include "feat/harris/harris.h"

#include "common/wall_time.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

DEFINE_string(image_path, "../feat/data/euroc01.png", "image file path");
DEFINE_int32(max_kps_num, 500, "max detector features num");
DEFINE_double(kps_min_distance, 30, "min kp distance");
DEFINE_double(kps_quality_level, 0.01, "qualityLevel of kp");

cv::Mat openCVMethod(const cv::Mat& image, int max_kps_num, double kps_min_distance, double kps_quality_level)
{
  std::vector<cv::Point2f> kps;
  cv::Mat draw_image;
  double start_time = com::WallTimeInSeconds();
  cv::goodFeaturesToTrack(image, kps, max_kps_num, kps_quality_level, kps_min_distance, cv::Mat());
  double cost_time = com::WallTimeInSeconds() - start_time;
  LOG(INFO) << "opencv detect " << kps.size() << " points, cost time: " << cost_time << " s";
  cv::cvtColor(image, draw_image, cv::COLOR_GRAY2BGR);
  for (const auto& pt : kps)
  {
    auto x = pt.x;
    auto y = pt.y;
    cv::circle(draw_image, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 0), -1);
  }
  cv::putText(draw_image, "", cv::Point2i(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
  return draw_image;
}

cv::Mat harrisDetector(const cv::Mat& image, int max_kps_num, double kps_min_distance, double kps_quality_level)
{
  std::vector<cv::Point2f> kps;
  cv::Mat draw_image;
  double start_time = com::WallTimeInSeconds();
  feat::goodFeaturesToTrack(image, kps, max_kps_num, kps_quality_level, kps_min_distance, cv::Mat());
  double cost_time = com::WallTimeInSeconds() - start_time;
  LOG(INFO) << "self detect " << kps.size() << " points, cost time: " << cost_time << " s";
  cv::cvtColor(image, draw_image, cv::COLOR_GRAY2BGR);
  for (const auto& pt : kps)
  {
    auto x = pt.x;
    auto y = pt.y;
    cv::circle(draw_image, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 0), -1);
  }
  cv::putText(draw_image, "", cv::Point2i(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
  return draw_image;
}

int main(int argc, char** argv)
{
  ::google::InitGoogleLogging("harris_test");
  FLAGS_stderrthreshold = google::INFO;
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat gray_image = cv::imread(FLAGS_image_path, cv::IMREAD_GRAYSCALE);

  int max_kps_num = FLAGS_max_kps_num;
  double kps_min_distance = FLAGS_kps_min_distance;
  double kps_quality_level = FLAGS_kps_quality_level;

  LOG_ASSERT(gray_image.empty() == false) << " Image is empty, please check you image path.";

  cv::Mat opencv_harris = openCVMethod(gray_image, max_kps_num, kps_min_distance, kps_quality_level);
  cv::Mat self_harris = harrisDetector(gray_image, max_kps_num, kps_min_distance, kps_quality_level);

  cv::imshow("opencv_harris", opencv_harris);
  cv::imshow("self_harris", self_harris);
  cv::waitKey(0);

  return 0;
}