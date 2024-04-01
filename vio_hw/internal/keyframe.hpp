#ifndef VIO_HELLO_WORLD_KEYFRAME_HPP
#define VIO_HELLO_WORLD_KEYFRAME_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

namespace viohw {
struct Keyframe
{
  int kfid_;
  cv::Mat imleft_, imright_;
  cv::Mat imleftraw_, imrightraw_;
  std::vector<cv::Mat> vpyr_imleft_, vpyr_imright_;
  bool is_stereo_;

  Keyframe() : kfid_(-1), is_stereo_(false) {}

  Keyframe(int kfid, const cv::Mat &imleftraw)
      : kfid_(kfid), imleftraw_(imleftraw.clone()), is_stereo_(false) {}

  Keyframe(int kfid, const cv::Mat &imleftraw, const cv::Mat &imright_raw)
      : kfid_(kfid), imleftraw_(imleftraw.clone()), imrightraw_(imright_raw.clone()), is_stereo_(true) {}

  Keyframe(int kfid, const cv::Mat &imleftraw, const std::vector<cv::Mat> &vpyrleft,
           const std::vector<cv::Mat> &vpyrright)
      : kfid_(kfid),
        imleftraw_(imleftraw.clone()),
        vpyr_imleft_(vpyrleft),
        vpyr_imright_(vpyrright),
        is_stereo_(true) {}

  Keyframe(int kfid, const cv::Mat &imleftraw, const cv::Mat &imrightraw,
           const std::vector<cv::Mat> &vpyrleft)
      : kfid_(kfid),
        imleftraw_(imleftraw.clone()),
        imrightraw_(imrightraw.clone()),
        vpyr_imleft_(vpyrleft) {}

  void displayInfo() {
    std::cout << "\n\n Keyframe struct object !  Info : id #" << kfid_
              << " - is stereo : " << is_stereo_;
    std::cout << " - imleft size : " << imleft_.size << " - imright size : " << imright_.size;
    std::cout << " - pyr left size : " << vpyr_imleft_.size()
              << " - pyr right size : " << vpyr_imright_.size() << "\n\n";
  }

  void releaseImages() {
    imleft_.release();
    imright_.release();
    imleftraw_.release();
    imrightraw_.release();
    vpyr_imleft_.clear();
    vpyr_imright_.clear();
  }
};

}  // namespace viohw

#endif  // VIO_HELLO_WORLD_KEYFRAME_HPP
