#ifndef DFM_PARAMETER_HPP
#define DFM_PARAMETER_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace dfm {

struct SuperPointConfig
{
  int32_t batchSize{ 1 };  //!< Number of inputs in a batch
  int32_t dlaCore{ -1 };   //!< Specify the DLA core to run network on.
  bool int8{ false };      //!< Allow runnning the network in Int8 mode.
  bool fp16{ false };      //!< Allow running the network in FP16 mode.
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;
  std::string onnxFilePath;
  std::string engineFilePath;
  float keypointThreshold;
  int maxKeypoints;
  int border;
};

struct LightGlueConfig
{
  int32_t batchSize{ 1 };
  int32_t dlaCore{ -1 };
  bool int8{ false };
  bool fp16{ false };
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;
  std::string onnxFilePath;
  std::string engineFilePath;
  int32_t imageWidth;
  int32_t imageHeight;
  float minMatcherScore;
};

inline bool readSpLgParameter( std::string config_file, SuperPointConfig &spConfig,
                               LightGlueConfig &lgConfig ) {
  const cv::FileStorage node( config_file.c_str(), cv::FileStorage::READ );
  if ( !node.isOpened() ) {
    std::cerr << "Config file open failed\n";
    exit( -1 );
  }

  spConfig.batchSize = node["SP.batchSize"];
  spConfig.dlaCore = node["SP.dlaCore"];
  spConfig.int8 = static_cast<int>( node["SP.int8"] );
  spConfig.fp16 = static_cast<int>( node["SP.fp16"] );
  node["SP.onnxFilePath"] >> spConfig.onnxFilePath;
  node["SP.engineFilePath"] >> spConfig.engineFilePath;
  spConfig.keypointThreshold = node["SP.keypointThreshold"];
  spConfig.maxKeypoints = node["SP.maxKeypoints"];
  spConfig.border = node["SP.border"];
  cv::FileNode SPinputTensorName = node["SP.inputTensorNames"];
  cv::FileNode SPoutputTensorName = node["SP.outputTensorNames"];
  spConfig.inputTensorNames.emplace_back( static_cast<std::string>( SPinputTensorName[0] ) );
  spConfig.outputTensorNames.emplace_back( static_cast<std::string>( SPoutputTensorName[0] ) );
  spConfig.outputTensorNames.emplace_back( static_cast<std::string>( SPoutputTensorName[1] ) );

  lgConfig.batchSize = node["LG.batchSize"];
  lgConfig.dlaCore = node["LG.dlaCore"];
  lgConfig.int8 = static_cast<int>( node["LG.int8"] );
  lgConfig.fp16 = static_cast<int>( node["LG.fp16"] );
  lgConfig.imageHeight = node["LG.imageHeight"];
  lgConfig.imageWidth = node["LG.imageWidth"];
  lgConfig.minMatcherScore = node["LG.minMatcherScore"];
  node["LG.onnxFilePath"] >> lgConfig.onnxFilePath;
  node["LG.engineFilePath"] >> lgConfig.engineFilePath;

  cv::FileNode LGinputTensorName = node["LG.inputTensorNames"];
  cv::FileNode LGoutputTensorName = node["LG.outputTensorNames"];
  lgConfig.inputTensorNames.emplace_back( static_cast<std::string>( LGinputTensorName[0] ) );
  lgConfig.inputTensorNames.emplace_back( static_cast<std::string>( LGinputTensorName[1] ) );
  lgConfig.inputTensorNames.emplace_back( static_cast<std::string>( LGinputTensorName[2] ) );
  lgConfig.inputTensorNames.emplace_back( static_cast<std::string>( LGinputTensorName[3] ) );
  lgConfig.outputTensorNames.emplace_back( static_cast<std::string>( LGoutputTensorName[0] ) );

  return true;
}

}  // namespace dfm

#endif  // DFM_PARAMETER_HPP