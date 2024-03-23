#pragma once
#include <float.h>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video.hpp"
namespace feat {

typedef short deriv_type;

struct ScharrDerivInvoker : cv::ParallelLoopBody
{
  ScharrDerivInvoker(const cv::Mat& _src, const cv::Mat& _dst)
      : src(_src), dst(_dst) {}

  void operator()(const cv::Range& range) const CV_OVERRIDE;

  const cv::Mat& src;
  const cv::Mat& dst;
};

struct LKTrackerInvoker : cv::ParallelLoopBody
{
  LKTrackerInvoker(const cv::Mat& _prevImg, const cv::Mat& _prevDeriv,
                   const cv::Mat& _nextImg, const cv::Point2f* _prevPts,
                   cv::Point2f* _nextPts, uchar* _status, float* _err,
                   cv::Size _winSize, cv::TermCriteria _criteria, int _level,
                   int _maxLevel, int _flags, float _minEigThreshold);

  void operator()(const cv::Range& range) const CV_OVERRIDE;

  const cv::Mat* prevImg;
  const cv::Mat* nextImg;
  const cv::Mat* prevDeriv;
  const cv::Point2f* prevPts;
  cv::Point2f* nextPts;
  uchar* status;
  float* err;
  cv::Size winSize;
  cv::TermCriteria criteria;
  int level;
  int maxLevel;
  int flags;
  float minEigThreshold;
};

class SparseOpticalFlow
{
 public:
  virtual void calc(cv::InputArray prevImg, cv::InputArray nextImg,
                    cv::InputArray prevPts, cv::InputOutputArray nextPts,
                    cv::OutputArray status,
                    cv::OutputArray err = cv::noArray()) = 0;
};

class SparsePyrLKOpticalFlow : public SparseOpticalFlow
{
 public:
  virtual cv::Size getWinSize() const = 0;
  virtual void setWinSize(cv::Size winSize) = 0;

  virtual int getMaxLevel() const = 0;
  virtual void setMaxLevel(int maxLevel) = 0;

  virtual cv::TermCriteria getTermCriteria() const = 0;
  virtual void setTermCriteria(cv::TermCriteria& crit) = 0;

  virtual int getFlags() const = 0;
  virtual void setFlags(int flags) = 0;

  virtual double getMinEigThreshold() const = 0;
  virtual void setMinEigThreshold(double minEigThreshold) = 0;

  static cv::Ptr<SparsePyrLKOpticalFlow> create(
      cv::Size winSize = cv::Size(21, 21), int maxLevel = 3,
      cv::TermCriteria crit = cv::TermCriteria(cv::TermCriteria::COUNT +
                                                   cv::TermCriteria::EPS,
                                               30, 0.01),
      int flags = 0, double minEigThreshold = 1e-4);
};

int buildOpticalFlowPyramid(cv::InputArray img, cv::OutputArrayOfArrays pyramid,
                            cv::Size winSize, int maxLevel,
                            bool withDerivatives = true,
                            int pyrBorder = cv::BORDER_REFLECT_101,
                            int derivBorder = cv::BORDER_CONSTANT,
                            bool tryReuseInputImage = true);
void calcOpticalFlowPyrLK(cv::InputArray prevImg, cv::InputArray nextImg,
                          cv::InputArray prevPts, cv::InputOutputArray nextPts,
                          cv::OutputArray status, cv::OutputArray err,
                          cv::Size winSize = cv::Size(21, 21), int maxLevel = 3,
                          cv::TermCriteria criteria = cv::TermCriteria(
                              cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                              30, 0.01),
                          int flags = 0, double minEigThreshold = 1e-4);
}  // namespace feat
