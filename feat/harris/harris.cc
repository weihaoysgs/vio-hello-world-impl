#include "feat/harris/harris.h"

namespace feat {
enum
{
  MINEIGENVAL = 0,
  HARRIS = 1,
  EIGENVALSVECS = 2
};

struct greaterThanPtr
{
  bool operator()(const float* a, const float* b) const
  // Ensure a fully deterministic result of the sort
  {
    return (*a > *b) ? true : (*a < *b) ? false : (a > b);
  }
};
static void calcMinEigenVal(const cv::Mat& _cov, cv::Mat& _dst)
{
  int i, j;
  cv::Size size = _cov.size();

  if (_cov.isContinuous() && _dst.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }

  for (i = 0; i < size.height; i++)
  {
    const float* cov = _cov.ptr<float>(i);
    float* dst = _dst.ptr<float>(i);

    j = 0;

#if CV_SIMD128
    {
      v_float32x4 half = v_setall_f32(0.5f);
      for (; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes)
      {
        v_float32x4 v_a, v_b, v_c, v_t;
        v_load_deinterleave(cov + j * 3, v_a, v_b, v_c);
        v_a *= half;
        v_c *= half;
        v_t = v_a - v_c;
        v_t = v_muladd(v_b, v_b, (v_t * v_t));
        v_store(dst + j, (v_a + v_c) - v_sqrt(v_t));
      }
    }
#endif  // CV_SIMD128

    for (; j < size.width; j++)
    {
      float a = cov[j * 3] * 0.5f;
      float b = cov[j * 3 + 1];
      float c = cov[j * 3 + 2] * 0.5f;
      dst[j] = (float)((a + c) - std::sqrt((a - c) * (a - c) + b * b));
    }
  }
}

static void calcHarris(const cv::Mat& _cov, cv::Mat& _dst, double k)
{
  int i, j;
  cv::Size size = _cov.size();

  if (_cov.isContinuous() && _dst.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }

  for (i = 0; i < size.height; i++)
  {
    const float* cov = _cov.ptr<float>(i);
    float* dst = _dst.ptr<float>(i);

    j = 0;

#if CV_SIMD128
    {
      v_float32x4 v_k = v_setall_f32((float)k);

      for (; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes)
      {
        v_float32x4 v_a, v_b, v_c;
        v_load_deinterleave(cov + j * 3, v_a, v_b, v_c);

        v_float32x4 v_ac_bb = v_a * v_c - v_b * v_b;
        v_float32x4 v_ac = v_a + v_c;
        v_float32x4 v_dst = v_ac_bb - v_k * v_ac * v_ac;
        v_store(dst + j, v_dst);
      }
    }
#endif  // CV_SIMD128

    for (; j < size.width; j++)
    {
      float a = cov[j * 3];
      float b = cov[j * 3 + 1];
      float c = cov[j * 3 + 2];
      dst[j] = (float)(a * c - b * b - k * (a + c) * (a + c));
    }
  }
}

static void eigen2x2(const float* cov, float* dst, int n)
{
  for (int j = 0; j < n; j++)
  {
    double a = cov[j * 3];
    double b = cov[j * 3 + 1];
    double c = cov[j * 3 + 2];

    double u = (a + c) * 0.5;
    double v = std::sqrt((a - c) * (a - c) * 0.25 + b * b);
    double l1 = u + v;
    double l2 = u - v;

    double x = b;
    double y = l1 - a;
    double e = fabs(x);

    if (e + fabs(y) < 1e-4)
    {
      y = b;
      x = l1 - c;
      e = fabs(x);
      if (e + fabs(y) < 1e-4)
      {
        e = 1. / (e + fabs(y) + FLT_EPSILON);
        x *= e, y *= e;
      }
    }

    double d = 1. / std::sqrt(x * x + y * y + DBL_EPSILON);
    dst[6 * j] = (float)l1;
    dst[6 * j + 2] = (float)(x * d);
    dst[6 * j + 3] = (float)(y * d);

    x = b;
    y = l2 - a;
    e = fabs(x);

    if (e + fabs(y) < 1e-4)
    {
      y = b;
      x = l2 - c;
      e = fabs(x);
      if (e + fabs(y) < 1e-4)
      {
        e = 1. / (e + fabs(y) + FLT_EPSILON);
        x *= e, y *= e;
      }
    }

    d = 1. / std::sqrt(x * x + y * y + DBL_EPSILON);
    dst[6 * j + 1] = (float)l2;
    dst[6 * j + 4] = (float)(x * d);
    dst[6 * j + 5] = (float)(y * d);
  }
}

static void calcEigenValsVecs(const cv::Mat& _cov, cv::Mat& _dst)
{
  cv::Size size = _cov.size();
  if (_cov.isContinuous() && _dst.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }

  for (int i = 0; i < size.height; i++)
  {
    const float* cov = _cov.ptr<float>(i);
    float* dst = _dst.ptr<float>(i);

    eigen2x2(cov, dst, size.width);
  }
}

static void cornerEigenValsVecs(const cv::Mat& src, cv::Mat& eigenv, int block_size, int aperture_size, int op_type, double k = 0.,
                                int borderType = cv::BORDER_DEFAULT)
{
  int depth = src.depth();
  double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
  if (aperture_size < 0) scale *= 2.0;
  if (depth == CV_8U) scale *= 255.0;
  scale = 1.0 / scale;

  CV_Assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);

  cv::Mat Dx, Dy;
  if (aperture_size > 0)
  {
    cv::Sobel(src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType);
    cv::Sobel(src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType);
  }
  else
  {
    cv::Scharr(src, Dx, CV_32F, 1, 0, scale, 0, borderType);
    cv::Scharr(src, Dy, CV_32F, 0, 1, scale, 0, borderType);
  }

  cv::Size size = src.size();
  cv::Mat cov(size, CV_32FC3);
  int i, j;

  for (i = 0; i < size.height; i++)
  {
    float* cov_data = cov.ptr<float>(i);
    const float* dxdata = Dx.ptr<float>(i);
    const float* dydata = Dy.ptr<float>(i);

#if CV_TRY_AVX
    if (haveAvx)
      j = cornerEigenValsVecsLine_AVX(dxdata, dydata, cov_data, size.width);
    else
#endif  // CV_TRY_AVX
      j = 0;

#if CV_SIMD128
    {
      for (; j <= size.width - v_float32x4::nlanes; j += v_float32x4::nlanes)
      {
        v_float32x4 v_dx = v_load(dxdata + j);
        v_float32x4 v_dy = v_load(dydata + j);

        v_float32x4 v_dst0, v_dst1, v_dst2;
        v_dst0 = v_dx * v_dx;
        v_dst1 = v_dx * v_dy;
        v_dst2 = v_dy * v_dy;

        v_store_interleave(cov_data + j * 3, v_dst0, v_dst1, v_dst2);
      }
    }
#endif  // CV_SIMD128

    for (; j < size.width; j++)
    {
      float dx = dxdata[j];
      float dy = dydata[j];

      cov_data[j * 3] = dx * dx;
      cov_data[j * 3 + 1] = dx * dy;
      cov_data[j * 3 + 2] = dy * dy;
    }
  }

  cv::boxFilter(cov, cov, cov.depth(), cv::Size(block_size, block_size), cv::Point(-1, -1), false, borderType);

  if (op_type == MINEIGENVAL)
    feat::calcMinEigenVal(cov, eigenv);
  else if (op_type == HARRIS)
    feat::calcHarris(cov, eigenv, k);
  else if (op_type == EIGENVALSVECS)
    feat::calcEigenValsVecs(cov, eigenv);
}

void cornerHarris(cv::InputArray _src, cv::OutputArray _dst, int blockSize, int ksize, double k, int borderType)
{
  cv::Mat src = _src.getMat();
  _dst.create(src.size(), CV_32FC1);
  cv::Mat dst = _dst.getMat();

  feat::cornerEigenValsVecs(src, dst, blockSize, ksize, HARRIS, k, borderType);
}

void cornerMinEigenVal(cv::InputArray _src, cv::OutputArray _dst, int blockSize, int ksize, int borderType)
{
  cv::Mat src = _src.getMat();
  _dst.create(src.size(), CV_32FC1);
  cv::Mat dst = _dst.getMat();

  feat::cornerEigenValsVecs(src, dst, blockSize, ksize, MINEIGENVAL, 0, borderType);
}

void goodFeaturesToTrack(cv::InputArray _image, cv::OutputArray _corners, int maxCorners, double qualityLevel, double minDistance,
                         cv::InputArray _mask, int blockSize, int gradientSize, bool useHarrisDetector, double harrisK)
{
  CV_Assert(qualityLevel > 0 && minDistance >= 0 && maxCorners >= 0);
  CV_Assert(_mask.empty() || (_mask.type() == CV_8UC1 && _mask.sameSize(_image)));

  cv::Mat image = _image.getMat(), eig, tmp;
  if (image.empty())
  {
    _corners.release();
    return;
  }

  if (useHarrisDetector)
    feat::cornerHarris(image, eig, blockSize, gradientSize, harrisK);
  else
    feat::cornerMinEigenVal(image, eig, blockSize, gradientSize);

  double maxVal = 0;
  cv::minMaxLoc(eig, 0, &maxVal, 0, 0, _mask);
  cv::threshold(eig, eig, maxVal * qualityLevel, 0, cv::THRESH_TOZERO);
  cv::dilate(eig, tmp, cv::Mat());

  cv::Size imgsize = image.size();
  std::vector<const float*> tmpCorners;

  // collect list of pointers to features - put them into temporary image
  cv::Mat mask = _mask.getMat();
  for (int y = 1; y < imgsize.height - 1; y++)
  {
    const float* eig_data = (const float*)eig.ptr(y);
    const float* tmp_data = (const float*)tmp.ptr(y);
    const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

    for (int x = 1; x < imgsize.width - 1; x++)
    {
      float val = eig_data[x];
      if (val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x])) tmpCorners.push_back(eig_data + x);
    }
  }

  std::vector<cv::Point2f> corners;
  size_t i, j, total = tmpCorners.size(), ncorners = 0;

  if (total == 0)
  {
    _corners.release();
    return;
  }

  std::sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

  if (minDistance >= 1)
  {
    // Partition the image into larger grids
    int w = image.cols;
    int h = image.rows;

    const int cell_size = cvRound(minDistance);
    const int grid_width = (w + cell_size - 1) / cell_size;
    const int grid_height = (h + cell_size - 1) / cell_size;

    std::vector<std::vector<cv::Point2f> > grid(grid_width * grid_height);

    minDistance *= minDistance;

    for (i = 0; i < total; i++)
    {
      int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
      int y = (int)(ofs / eig.step);
      int x = (int)((ofs - y * eig.step) / sizeof(float));

      bool good = true;

      int x_cell = x / cell_size;
      int y_cell = y / cell_size;

      int x1 = x_cell - 1;
      int y1 = y_cell - 1;
      int x2 = x_cell + 1;
      int y2 = y_cell + 1;

      // boundary check
      x1 = std::max(0, x1);
      y1 = std::max(0, y1);
      x2 = std::min(grid_width - 1, x2);
      y2 = std::min(grid_height - 1, y2);

      for (int yy = y1; yy <= y2; yy++)
      {
        for (int xx = x1; xx <= x2; xx++)
        {
          std::vector<cv::Point2f>& m = grid[yy * grid_width + xx];

          if (m.size())
          {
            for (j = 0; j < m.size(); j++)
            {
              float dx = x - m[j].x;
              float dy = y - m[j].y;

              if (dx * dx + dy * dy < minDistance)
              {
                good = false;
                goto break_out;
              }
            }
          }
        }
      }

    break_out:

      if (good)
      {
        grid[y_cell * grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

        corners.push_back(cv::Point2f((float)x, (float)y));
        ++ncorners;

        if (maxCorners > 0 && (int)ncorners == maxCorners) break;
      }
    }
  }
  else
  {
    for (i = 0; i < total; i++)
    {
      int ofs = (int)((const uchar*)tmpCorners[i] - eig.ptr());
      int y = (int)(ofs / eig.step);
      int x = (int)((ofs - y * eig.step) / sizeof(float));

      corners.push_back(cv::Point2f((float)x, (float)y));
      ++ncorners;
      if (maxCorners > 0 && (int)ncorners == maxCorners) break;
    }
  }

  cv::Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);
}

}  // namespace feat