#include "dfm/internal/lightglue.hpp"

namespace dfm {

LightGlue::LightGlue( const LightGlueConfig &config ) : lgConfig( std::move( config ) ) {
  trtLogger::setReportableSeverity( trtLogger::Logger::Severity::kERROR );
}

bool LightGlue::build() {
  if ( deserializeEngine() ) {
    return true;
  }

  auto builder = trtCommon::trtUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder( sample::gLogger.getTRTLogger() ) );
  if ( !builder ) {
    return false;
  }

  const auto explicitBatch =
      1U << static_cast<uint32_t>( nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH );
  auto network = trtCommon::trtUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2( explicitBatch ) );
  if ( !network ) {
    return false;
  }

  auto config = trtCommon::trtUniquePtr<nvinfer1::IBuilderConfig>( builder->createBuilderConfig() );
  if ( !config ) {
    return false;
  }

  auto parser = trtCommon::trtUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser( *network, sample::gLogger.getTRTLogger() ) );
  if ( !parser ) {
    return false;
  }
  auto profile = builder->createOptimizationProfile();
  if ( !profile ) {
    return false;
  }

  profile->setDimensions( lgConfig.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN,
                          nvinfer1::Dims3( 1, 1, 2 ) );
  profile->setDimensions( lgConfig.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::Dims3( 1, 500, 2 ) );
  profile->setDimensions( lgConfig.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX,
                          nvinfer1::Dims3( 1, 3000, 2 ) );

  profile->setDimensions( lgConfig.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMIN,
                          nvinfer1::Dims3( 1, 1, 2 ) );
  profile->setDimensions( lgConfig.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::Dims3( 1, 500, 2 ) );
  profile->setDimensions( lgConfig.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMAX,
                          nvinfer1::Dims3( 1, 3000, 2 ) );

  profile->setDimensions( lgConfig.inputTensorNames[2].c_str(), nvinfer1::OptProfileSelector::kMIN,
                          nvinfer1::Dims3( 1, 1, 256 ) );
  profile->setDimensions( lgConfig.inputTensorNames[2].c_str(), nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::Dims3( 1, 500, 256 ) );
  profile->setDimensions( lgConfig.inputTensorNames[2].c_str(), nvinfer1::OptProfileSelector::kMAX,
                          nvinfer1::Dims3( 1, 3000, 256 ) );

  profile->setDimensions( lgConfig.inputTensorNames[3].c_str(), nvinfer1::OptProfileSelector::kMIN,
                          nvinfer1::Dims3( 1, 1, 256 ) );
  profile->setDimensions( lgConfig.inputTensorNames[3].c_str(), nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::Dims3( 1, 500, 256 ) );
  profile->setDimensions( lgConfig.inputTensorNames[3].c_str(), nvinfer1::OptProfileSelector::kMAX,
                          nvinfer1::Dims3( 1, 3000, 256 ) );

  // profile->setDimensions(lgConfig.outputTensorNames[0].c_str(),
  // nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 2));
  // profile->setDimensions(lgConfig.outputTensorNames[0].c_str(),
  // nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(500, 2));
  // profile->setDimensions(lgConfig.outputTensorNames[0].c_str(),
  // nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(3000, 2));

  // profile->setDimensions(lgConfig.outputTensorNames[1].c_str(),
  // nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, 1});
  // profile->setDimensions(lgConfig.outputTensorNames[1].c_str(),
  // nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, 500});
  // profile->setDimensions(lgConfig.outputTensorNames[1].c_str(),
  // nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, 3000});

  config->addOptimizationProfile( profile );

  auto constructed = constructNetwork( builder, network, config, parser );
  if ( !constructed ) {
    return false;
  }

  // CUDA stream used for profiling by the builder.
  auto profileStream = samplesCommon::makeCudaStream();
  if ( !profileStream ) {
    return false;
  }
  config->setProfileStream( *profileStream );

  trtCommon::trtUniquePtr<nvinfer1::IHostMemory> plan{
      builder->buildSerializedNetwork( *network, *config ) };
  if ( !plan ) {
    return false;
  }

  mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime( sample::gLogger.getTRTLogger() ) );
  if ( !mRuntime ) {
    return false;
  }

  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
      mRuntime->deserializeCudaEngine( plan->data(), plan->size() ),
      samplesCommon::InferDeleter() );
  if ( !mEngine ) {
    return false;
  }

  saveEngine();

  return true;
}

bool LightGlue::constructNetwork( trtCommon::trtUniquePtr<nvinfer1::IBuilder> &builder,
                                  trtCommon::trtUniquePtr<nvinfer1::INetworkDefinition> &network,
                                  trtCommon::trtUniquePtr<nvinfer1::IBuilderConfig> &config,
                                  trtCommon::trtUniquePtr<nvonnxparser::IParser> &parser ) {
  auto parsed = parser->parseFromFile(
      lgConfig.onnxFilePath.c_str(), static_cast<int>( sample::gLogger.getReportableSeverity() ) );
  if ( !parsed ) {
    return false;
  }

  if ( lgConfig.fp16 ) {
    config->setFlag( nvinfer1::BuilderFlag::kFP16 );
  }
  if ( lgConfig.int8 ) {
    config->setFlag( nvinfer1::BuilderFlag::kINT8 );
    samplesCommon::setAllDynamicRanges( network.get(), 127.0F, 127.0F );
  }

  samplesCommon::enableDLA( builder.get(), config.get(), lgConfig.dlaCore );

  return true;
}

bool LightGlue::deserializeEngine() {
  std::ifstream file( lgConfig.engineFilePath.c_str(), std::ios::binary );
  if ( file.is_open() ) {
    file.seekg( 0, std::ifstream::end );
    size_t size = file.tellg();
    file.seekg( 0, std::ifstream::beg );
    char *model_stream = new char[size];
    file.read( model_stream, size );
    file.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime( trtLogger::gLogger );
    if ( runtime == nullptr ) {
      delete[] model_stream;
      return false;
    }
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine( model_stream, size ) );
    if ( mEngine == nullptr ) {
      delete[] model_stream;
      return false;
    }
    delete[] model_stream;
    return true;
  }
  return false;
}

void LightGlue::saveEngine() {
  if ( lgConfig.engineFilePath.empty() ) return;
  if ( mEngine != nullptr ) {
    nvinfer1::IHostMemory *data = mEngine->serialize();
    std::ofstream file( lgConfig.engineFilePath, std::ios::binary );
    if ( !file ) return;
    file.write( reinterpret_cast<const char *>( data->data() ), data->size() );
    trtLogger::gLogInfo << "Engine file save into: " << lgConfig.engineFilePath << std::endl;
  }
}

bool LightGlue::processInput( const trtCommon::BufferManager &buffers,
                              Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                              Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 ) {
  auto *keypoints_0_buffer =
      static_cast<float *>( buffers.getHostBuffer( lgConfig.inputTensorNames[0] ) );
  auto *keypoints_1_buffer =
      static_cast<float *>( buffers.getHostBuffer( lgConfig.inputTensorNames[1] ) );
  auto *descriptors_0_buffer =
      static_cast<float *>( buffers.getHostBuffer( lgConfig.inputTensorNames[2] ) );
  auto *descriptors_1_buffer =
      static_cast<float *>( buffers.getHostBuffer( lgConfig.inputTensorNames[3] ) );

  for ( int colk0 = 0; colk0 < features0.cols(); ++colk0 ) {
    for ( int rowk0 = 1; rowk0 < 3; ++rowk0 ) {
      keypoints_0_buffer[colk0 * 2 + ( rowk0 - 1 )] = features0( rowk0, colk0 );
    }
  }

  // for (int rowd0 = 3; rowd0 < features0.rows(); ++rowd0)
  // {
  //   for (int cold0 = 0; cold0 < features0.cols(); ++cold0)
  //   {
  //     descriptors_0_buffer[(rowd0 - 3) * features0.cols() + cold0] = features0(rowd0, cold0);
  //   }
  // }
  for ( int cold0 = 0; cold0 < features0.cols(); ++cold0 ) {
    for ( int rowd0 = 3; rowd0 < features0.rows(); ++rowd0 ) {
      descriptors_0_buffer[cold0 * 256 + ( rowd0 - 3 )] = features0( rowd0, cold0 );
    }
  }

  for ( int colk1 = 0; colk1 < features1.cols(); ++colk1 ) {
    for ( int rowk1 = 1; rowk1 < 3; ++rowk1 ) {
      keypoints_1_buffer[colk1 * 2 + ( rowk1 - 1 )] = features1( rowk1, colk1 );
    }
  }

  // for (int rowd1 = 3; rowd1 < features1.rows(); ++rowd1)
  // {
  //   for (int cold1 = 0; cold1 < features1.cols(); ++cold1)
  //   {
  //     descriptors_1_buffer[(rowd1 - 3) * features1.cols() + cold1] = features1(rowd1, cold1);
  //   }
  // }
  for ( int cold1 = 0; cold1 < features1.cols(); ++cold1 ) {
    for ( int rowd1 = 3; rowd1 < features1.rows(); ++rowd1 ) {
      descriptors_1_buffer[cold1 * 256 + ( rowd1 - 3 )] = features1( rowd1, cold1 );
    }
  }
  return true;
}

bool LightGlue::infer( Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                       Eigen::Matrix<double, 259, Eigen::Dynamic> &features1 ) {
  if ( mContext == nullptr ) {
    mContext =
        trtCommon::trtUniquePtr<nvinfer1::IExecutionContext>( mEngine->createExecutionContext() );
    if ( !mContext ) {
      return false;
    }
  }

  // assert(mEngine->getNbBindings() == 6);

  const int keypoints_0_index = mEngine->getBindingIndex( lgConfig.inputTensorNames[0].c_str() );
  const int keypoints_1_index = mEngine->getBindingIndex( lgConfig.inputTensorNames[1].c_str() );
  const int descriptors_0_index = mEngine->getBindingIndex( lgConfig.inputTensorNames[2].c_str() );
  const int descriptors_1_index = mEngine->getBindingIndex( lgConfig.inputTensorNames[3].c_str() );

  const int output_score0_index = mEngine->getBindingIndex( lgConfig.outputTensorNames[0].c_str() );

  mContext->setBindingDimensions( keypoints_0_index, nvinfer1::Dims3( 1, features0.cols(), 2 ) );
  mContext->setBindingDimensions( keypoints_1_index, nvinfer1::Dims3( 1, features1.cols(), 2 ) );
  mContext->setBindingDimensions( descriptors_0_index,
                                  nvinfer1::Dims3( 1, features0.cols(), 256 ) );
  mContext->setBindingDimensions( descriptors_1_index,
                                  nvinfer1::Dims3( 1, features1.cols(), 256 ) );

  nvinfer1::Dims out_score0_dims{ 1, features0.cols() };
  nvinfer1::Dims2 out_matches0_dims( features0.cols(), 2 );

  keypoints_0_dims_ = mContext->getBindingDimensions( keypoints_0_index );
  descriptors_0_dims_ = mContext->getBindingDimensions( descriptors_0_index );
  keypoints_1_dims_ = mContext->getBindingDimensions( keypoints_1_index );
  descriptors_1_dims_ = mContext->getBindingDimensions( descriptors_1_index );

  output_scores0_dims_ = mContext->getBindingDimensions( output_score0_index );
  // std::cout << output_scores0_dims_.d[0] << ", " << output_scores0_dims_.d[1] << "," <<
  // output_scores0_dims_.d[2] << std::endl;

  trtCommon::BufferManager buffers( mEngine, 0, mContext.get() );

  assert( lgConfig.inputTensorNames.size() == 4 );
  if ( !processInput( buffers, features0, features1 ) ) {
    return false;
  }

  buffers.copyInputToDevice();

  bool status = mContext->executeV2( buffers.getDeviceBindings().data() );
  if ( !status ) {
    return false;
  }

  buffers.copyOutputToHost();

  if ( !verifyOutput( buffers ) ) {
    return false;
  }

  return true;
}

std::vector<std::tuple<int, int, float>> findMaxInRowAndColumn( const Eigen::MatrixXf &matrix,
                                                                float threshold ) {
  const int rows = matrix.rows();
  const int cols = matrix.cols();
  Eigen::Matrix<float, Eigen::Dynamic, 1> maxValuesRow = matrix.rowwise().maxCoeff();
  Eigen::Matrix<float, 1, Eigen::Dynamic> maxValuesCol = matrix.colwise().maxCoeff();

  std::vector<std::tuple<int, int, float>> result;

  for ( int i = 0; i < matrix.rows(); ++i ) {
    for ( int j = 0; j < matrix.cols(); ++j ) {
      bool isMaxInRow = ( maxValuesRow( i ) == matrix( i, j ) );

      bool isMaxInColumn = ( maxValuesCol( 0, j ) == matrix( i, j ) );

      float conf = std::exp( matrix( i, j ) );

      if ( isMaxInRow && isMaxInColumn && ( conf > threshold ) ) {
        result.push_back( std::make_tuple( i, j, conf ) );
      }
    }
  }

  return result;
}

bool LightGlue::verifyOutput( const samplesCommon::BufferManager &buffers ) {
  float *output_score =
      static_cast<float *>( buffers.getHostBuffer( lgConfig.outputTensorNames[0] ) );
  int scores_map_h = output_scores0_dims_.d[1];
  int scores_map_w = output_scores0_dims_.d[2];
  std::vector<float> score( output_score, output_score + scores_map_h * scores_map_w );
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matrix(
      output_score, scores_map_h, scores_map_w );

  lgMatcherResult.clear();
  lgMatcherResult = findMaxInRowAndColumn( matrix, lgConfig.minMatcherScore );
  // saveLightGlueInferResultToDisk("../result/matcher.txt");
  return true;
}

void LightGlue::saveLightGlueInferResultToDisk( const std::string path, std::vector<float> score ) {
  std::ofstream fout( path );
  for ( int i = 0; i < score.size(); i++ ) {
    fout << score[i] << " ";
    if ( ( i + 1 ) % 1000 == 0 ) fout << "\n";
  }
}

bool LightGlue::matcher( Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                         Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                         std::vector<cv::DMatch> &matches, bool outlier_rejection ) {
  dfm::TimeLog timer;
  matches.clear();
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features0 =
      normalizeKeypoints( features0, lgConfig.imageWidth, lgConfig.imageHeight );
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features1 =
      normalizeKeypoints( features1, lgConfig.imageWidth, lgConfig.imageHeight );
  // dfm::SuperPoint::saveResultToDisk("../result/normal0.kps.txt", norm_features0);
  // dfm::SuperPoint::saveResultToDisk("../result/normal1.kps.txt", norm_features1);

  infer( norm_features0, norm_features1 );
  return true;
}

Eigen::Matrix<double, 259, Eigen::Dynamic> LightGlue::normalizeKeypoints(
    const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height ) {
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
  norm_features.resize( 259, features.cols() );
  norm_features = features;
  for ( int col = 0; col < features.cols(); ++col ) {
    norm_features( 1, col ) =
        ( features( 1, col ) - width / 2 ) / ( std::max( width, height ) * 0.5 );
    norm_features( 2, col ) =
        ( features( 2, col ) - height / 2 ) / ( std::max( width, height ) * 0.5 );
  }
  return norm_features;
}
}  // namespace dfm
