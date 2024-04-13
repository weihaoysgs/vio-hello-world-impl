#ifndef LIGHT_GLUE_HPP
#define LIGHT_GLUE_HPP

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "dfm/internal/parameter.hpp"
#include "dfm/internal/superpoint.hpp"
#include "dfm/internal/tensorrt_common.hpp"
#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/common.h"

namespace dfm {

class LightGlue
{
public:
  LightGlue() = default;

  ~LightGlue() = default;

  LightGlue( const LightGlueConfig &config );

  bool build();

  bool infer( Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
              Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 );
  bool matcher( Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                std::vector<cv::DMatch> &matches, bool outlier_rejection );

  bool constructNetwork( trtCommon::trtUniquePtr<nvinfer1::IBuilder> &builder,
                         trtCommon::trtUniquePtr<nvinfer1::INetworkDefinition> &network,
                         trtCommon::trtUniquePtr<nvinfer1::IBuilderConfig> &config,
                         trtCommon::trtUniquePtr<nvonnxparser::IParser> &parser );

  bool processInput( const trtCommon::BufferManager &buffers,
                     Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                     Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 );

  bool verifyOutput( const samplesCommon::BufferManager &buffers );

  Eigen::Matrix<double, 259, Eigen::Dynamic> normalizeKeypoints(
      const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height );

  bool deserializeEngine();

  void saveEngine();

  void saveLightGlueInferResultToDisk( const std::string save_path, std::vector<float> score );

  //! add mutex when running in a multithread
  std::vector<std::tuple<int, int, float>> getLGMatcherResult() const { return lgMatcherResult; }

private:
  std::vector<std::tuple<int, int, float>> lgMatcherResult;

  LightGlueConfig lgConfig;
  nvinfer1::Dims mInputDims;   //!< The dimensions of the input to the network.
  nvinfer1::Dims mOutputDims;  //!< The dimensions of the output to the network.

  std::shared_ptr<nvinfer1::IRuntime>
      mRuntime;  //!< The TensorRT runtime used to deserialize the engine
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;  //!< The TensorRT engine used to run the network
  std::shared_ptr<nvinfer1::IExecutionContext> mContext;

  nvinfer1::Dims keypoints_0_dims_{};
  nvinfer1::Dims descriptors_0_dims_{};
  nvinfer1::Dims keypoints_1_dims_{};
  nvinfer1::Dims descriptors_1_dims_{};
  nvinfer1::Dims output_scores0_dims_{};
  nvinfer1::Dims output_matcher0_dims_{};
};

}  // namespace dfm

#endif  // LIGHT_GLUE_HPP