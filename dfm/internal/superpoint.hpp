#ifndef SUPERPOINT_HPP
#define SUPERPOINT_HPP

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "dfm/internal/parameter.hpp"
#include "dfm/internal/tensorrt_common.hpp"
#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/common.h"

namespace dfm {

class SuperPoint
{
public:
  SuperPoint() = default;
  SuperPoint( const SuperPointConfig &config );

  ~SuperPoint() = default;

  bool build();

  bool infer( const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features );

  bool constructNetwork( trtCommon::trtUniquePtr<nvinfer1::IBuilder> &builder,
                         trtCommon::trtUniquePtr<nvinfer1::INetworkDefinition> &network,
                         trtCommon::trtUniquePtr<nvinfer1::IBuilderConfig> &config,
                         trtCommon::trtUniquePtr<nvonnxparser::IParser> &parser );

  bool processInput( const trtCommon::BufferManager &buffers, const cv::Mat &image );

  bool verifyOutput( const samplesCommon::BufferManager &buffers,
                     Eigen::Matrix<double, 259, Eigen::Dynamic> &features );

  void saveEngine();

  bool deserializeEngine();

  void findHighScoreIndex( std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                           int h, int w, double threshold );

  void removeBorders( std::vector<std::vector<int>> &keypoints, std::vector<float> &scores,
                      int border, int height, int width );

  std::vector<size_t> sortIndexes( std::vector<float> &data );

  void topKkeypoints( std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k );

  std::vector<std::vector<int>> getKeypoints() { return mKeypoints; }

  void sampleDescriptors( std::vector<std::vector<int>> &keypoints, float *descriptors,
                          std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w,
                          int s = 8 );

  static void saveResultToDisk( const std::string &save_path,
                                Eigen::Matrix<double, 259, Eigen::Dynamic> &features );

  void setMaxExtractorKpsNumber( int num ) { spConfig.maxKeypoints = num; }

private:
  SuperPointConfig spConfig;
  nvinfer1::Dims mInputDims;   //!< The dimensions of the input to the network.
  nvinfer1::Dims mOutputDims;  //!< The dimensions of the output to the network.
  nvinfer1::Dims mSemiDims;
  nvinfer1::Dims mDescDims;
  std::vector<std::vector<int>> mKeypoints;
  std::vector<std::vector<double>> mDescriptors;

  std::shared_ptr<nvinfer1::IRuntime>
      mRuntime;  //!< The TensorRT runtime used to deserialize the engine
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;  //!< The TensorRT engine used to run the network
};

}  // namespace dfm

#endif  // SUPERPOINT_HPP