#ifndef VIO_HELLO_WORLD_DRAW_UTILS_H
#define VIO_HELLO_WORLD_DRAW_UTILS_H

#include <algorithm>
#include <opencv2/opencv.hpp>

namespace com {

template <typename T>
cv::Mat DrawKeyPoints(const cv::Mat &image, std::vector<cv::Point_<T>> &kps,
                   const std::string &text = "")
{
  cv::Mat draw_image;
  if (image.channels() == 1)
  {
    cv::cvtColor(image, draw_image, cv::COLOR_GRAY2BGR);
  }
  else
  {
    draw_image = image.clone();
  }
  std::for_each(kps.begin(), kps.end(), [&](const cv::Point_<T> &kp) {
    cv::circle(draw_image, cv::Point2f(kp.x, kp.y), 2, cv::Scalar(0, 255, 0), -1);
  });
  if (!text.empty())
    cv::putText(draw_image, text, cv::Point2i(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(0, 0, 255));
  return draw_image;
}

inline cv::Mat DrawKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                          const std::string &text = "")
{
  std::vector<cv::Point2f> pts;
  std::for_each(kps.begin(), kps.end(), [&](const cv::KeyPoint &kp) { pts.emplace_back(kp.pt); });
  return DrawKeyPoints(image, pts, text);
}
}  // namespace com
#endif  // VIO_HELLO_WORLD_DRAW_UTILS_H
