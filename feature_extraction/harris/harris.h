#ifndef VIO_HELLO_WORLD_HARRIS_H
#define VIO_HELLO_WORLD_HARRIS_H

#include "glog/logging.h"
#include "opencv2/opencv.hpp"

namespace feat {

void goodFeaturesToTrack(cv::InputArray image, cv::OutputArray corners, int maxCorners, double qualityLevel, double minDistance,
                         cv::InputArray mask = cv::noArray(), int blockSize = 3, int gradientSize = 3, bool useHarrisDetector = false,
                         double k = 0.04);
void cornerMinEigenVal(cv::InputArray src, cv::OutputArray dst, int blockSize, int ksize = 3, int borderType = cv::BORDER_DEFAULT);
void cornerHarris(cv::InputArray src, cv::OutputArray dst, int blockSize, int ksize, double k, int borderType = cv::BORDER_DEFAULT);

}  // namespace feat

#endif  // VIO_HELLO_WORLD_HARRIS_H
