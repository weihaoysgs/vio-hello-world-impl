// Ref: https://github.com/sair-lab/AirVO

#include "dfm/internal/superpoint.hpp"

namespace dfm {

SuperPoint::SuperPoint( const SuperPointConfig &config ) : spConfig( std::move( config ) ) {
  trtLogger::setReportableSeverity( trtLogger::Logger::Severity::kINFO );
}

bool SuperPoint::build() {
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

  profile->setDimensions( spConfig.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN,
                          nvinfer1::Dims4( 1, 1, 100, 100 ) );
  profile->setDimensions( spConfig.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT,
                          nvinfer1::Dims4( 1, 1, 500, 500 ) );
  profile->setDimensions( spConfig.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX,
                          nvinfer1::Dims4( 1, 1, 1500, 1500 ) );

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

bool SuperPoint::constructNetwork( trtCommon::trtUniquePtr<nvinfer1::IBuilder> &builder,
                                   trtCommon::trtUniquePtr<nvinfer1::INetworkDefinition> &network,
                                   trtCommon::trtUniquePtr<nvinfer1::IBuilderConfig> &config,
                                   trtCommon::trtUniquePtr<nvonnxparser::IParser> &parser ) {
  auto parsed = parser->parseFromFile(
      spConfig.onnxFilePath.c_str(), static_cast<int>( sample::gLogger.getReportableSeverity() ) );
  if ( !parsed ) {
    return false;
  }

  if ( spConfig.fp16 ) {
    config->setFlag( nvinfer1::BuilderFlag::kFP16 );
  }
  if ( spConfig.int8 ) {
    config->setFlag( nvinfer1::BuilderFlag::kINT8 );
    samplesCommon::setAllDynamicRanges( network.get(), 127.0F, 127.0F );
  }

  samplesCommon::enableDLA( builder.get(), config.get(), spConfig.dlaCore );

  return true;
}

bool SuperPoint::deserializeEngine() {
  std::ifstream file( spConfig.engineFilePath.c_str(), std::ios::binary );
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

void SuperPoint::saveEngine() {
  if ( spConfig.engineFilePath.empty() ) return;
  if ( mEngine != nullptr ) {
    nvinfer1::IHostMemory *data = mEngine->serialize();
    std::ofstream file( spConfig.engineFilePath, std::ios::binary );
    if ( !file ) return;
    file.write( reinterpret_cast<const char *>( data->data() ), data->size() );
    trtLogger::gLogInfo << "Engine file save into: " << spConfig.engineFilePath << std::endl;
  }
}

bool SuperPoint::processInput( const trtCommon::BufferManager &buffers, const cv::Mat &image ) {
  mInputDims.d[2] = image.rows;
  mInputDims.d[3] = image.cols;
  mSemiDims.d[1] = image.rows;
  mSemiDims.d[2] = image.cols;
  mDescDims.d[1] = 256;
  mDescDims.d[2] = image.rows / 8;
  mDescDims.d[3] = image.cols / 8;
  assert( image.channels() == 1 );
  auto *host_data_buffer =
      static_cast<float *>( buffers.getHostBuffer( spConfig.inputTensorNames[0] ) );
  /// TODO parallel acceleration
  for ( int row = 0; row < image.rows; ++row ) {
    for ( int col = 0; col < image.cols; ++col ) {
      host_data_buffer[row * image.cols + col] =
          float( image.at<unsigned char>( row, col ) ) / 255.0;
    }
  }
  return true;
}

void SuperPoint::findHighScoreIndex( std::vector<float> &scores,
                                     std::vector<std::vector<int>> &keypoints, int h, int w,
                                     double threshold ) {
  std::vector<float> new_scores;
  for ( int i = 0; i < scores.size(); ++i ) {
    if ( scores[i] > threshold ) {
      std::vector<int> location = { int( i / w ), i % w };
      keypoints.emplace_back( location );
      new_scores.push_back( scores[i] );
    }
  }
  scores.swap( new_scores );
}

std::vector<size_t> SuperPoint::sortIndexes( std::vector<float> &data ) {
  std::vector<size_t> indexes( data.size() );
  iota( indexes.begin(), indexes.end(), 0 );
  sort( indexes.begin(), indexes.end(),
        [&data]( size_t i1, size_t i2 ) { return data[i1] > data[i2]; } );
  return indexes;
}

void SuperPoint::topKkeypoints( std::vector<std::vector<int>> &keypoints,
                                std::vector<float> &scores, int k ) {
  if ( k < keypoints.size() && k != -1 ) {
    std::vector<std::vector<int>> keypoints_top_k;
    std::vector<float> scores_top_k;
    std::vector<size_t> indexes = sortIndexes( scores );
    for ( int i = 0; i < k; ++i ) {
      keypoints_top_k.push_back( keypoints[indexes[i]] );
      scores_top_k.push_back( scores[indexes[i]] );
    }
    keypoints.swap( keypoints_top_k );
    scores.swap( scores_top_k );
  }
}

void SuperPoint::removeBorders( std::vector<std::vector<int>> &keypoints,
                                std::vector<float> &scores, int border, int height, int width ) {
  std::vector<std::vector<int>> keypoints_selected;
  std::vector<float> scores_selected;
  for ( int i = 0; i < keypoints.size(); ++i ) {
    bool flag_h = ( keypoints[i][0] >= border ) && ( keypoints[i][0] < ( height - border ) );
    bool flag_w = ( keypoints[i][1] >= border ) && ( keypoints[i][1] < ( width - border ) );
    if ( flag_h && flag_w ) {
      keypoints_selected.push_back( std::vector<int>{ keypoints[i][1], keypoints[i][0] } );
      scores_selected.push_back( scores[i] );
    }
  }
  keypoints.swap( keypoints_selected );
  scores.swap( scores_selected );
}

void normalizeKeypoints( const std::vector<std::vector<int>> &keypoints,
                         std::vector<std::vector<double>> &keypoints_norm, int h, int w, int s ) {
  for ( auto &keypoint : keypoints ) {
    std::vector<double> kp = { keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5 };
    kp[0] = kp[0] / ( w * s - s / 2 - 0.5 );
    kp[1] = kp[1] / ( h * s - s / 2 - 0.5 );
    kp[0] = kp[0] * 2 - 1;
    kp[1] = kp[1] * 2 - 1;
    keypoints_norm.push_back( kp );
  }
}

int clip( int val, int max ) {
  if ( val < 0 ) return 0;
  return std::min( val, max - 1 );
}

void gridSample( const float *input, std::vector<std::vector<double>> &grid,
                 std::vector<std::vector<double>> &output, int dim, int h, int w ) {
  // descriptors 1, 256, image_height/8, image_width/8
  // keypoints 1, 1, number, 2
  // out 1, 256, 1, number
  for ( auto &g : grid ) {
    double ix = ( ( g[0] + 1 ) / 2 ) * ( w - 1 );
    double iy = ( ( g[1] + 1 ) / 2 ) * ( h - 1 );

    int ix_nw = clip( std::floor( ix ), w );
    int iy_nw = clip( std::floor( iy ), h );

    int ix_ne = clip( ix_nw + 1, w );
    int iy_ne = clip( iy_nw, h );

    int ix_sw = clip( ix_nw, w );
    int iy_sw = clip( iy_nw + 1, h );

    int ix_se = clip( ix_nw + 1, w );
    int iy_se = clip( iy_nw + 1, h );

    double nw = ( ix_se - ix ) * ( iy_se - iy );
    double ne = ( ix - ix_sw ) * ( iy_sw - iy );
    double sw = ( ix_ne - ix ) * ( iy - iy_ne );
    double se = ( ix - ix_nw ) * ( iy - iy_nw );

    std::vector<double> descriptor;
    for ( int i = 0; i < dim; ++i ) {
      // 256x60x106 dhw
      // x * height * depth + y * depth + z
      float nw_val = input[i * h * w + iy_nw * w + ix_nw];
      float ne_val = input[i * h * w + iy_ne * w + ix_ne];
      float sw_val = input[i * h * w + iy_sw * w + ix_sw];
      float se_val = input[i * h * w + iy_se * w + ix_se];
      descriptor.push_back( nw_val * nw + ne_val * ne + sw_val * sw + se_val * se );
    }
    output.push_back( descriptor );
  }
}

template <typename Iter_T>
double vector_normalize( Iter_T first, Iter_T last ) {
  return sqrt( inner_product( first, last, first, 0.0 ) );
}

void normalizeDescriptors( std::vector<std::vector<double>> &dest_descriptors ) {
  for ( auto &descriptor : dest_descriptors ) {
    double norm_inv = 1.0 / vector_normalize( descriptor.begin(), descriptor.end() );
    std::transform( descriptor.begin(), descriptor.end(), descriptor.begin(),
                    std::bind1st( std::multiplies<double>(), norm_inv ) );
  }
}

void SuperPoint::sampleDescriptors( std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<std::vector<double>> &dest_descriptors, int dim,
                                    int h, int w, int s ) {
  std::vector<std::vector<double>> keypoints_norm;
  normalizeKeypoints( keypoints, keypoints_norm, h, w, s );
  gridSample( descriptors, keypoints_norm, dest_descriptors, dim, h, w );
  normalizeDescriptors( dest_descriptors );
}

bool SuperPoint::verifyOutput( const samplesCommon::BufferManager &buffers,
                               Eigen::Matrix<double, 259, Eigen::Dynamic> &features ) {
  mKeypoints.clear();
  mDescriptors.clear();
  auto *output_score =
      static_cast<float *>( buffers.getHostBuffer( spConfig.outputTensorNames[0] ) );
  auto *output_desc =
      static_cast<float *>( buffers.getHostBuffer( spConfig.outputTensorNames[1] ) );
  if ( output_desc == nullptr || output_score == nullptr ) {
    return false;
  }
  int semi_feature_map_h = mSemiDims.d[1];
  int semi_feature_map_w = mSemiDims.d[2];
  std::vector<float> scores_vec( output_score,
                                 output_score + semi_feature_map_h * semi_feature_map_w );
  findHighScoreIndex( scores_vec, mKeypoints, semi_feature_map_h, semi_feature_map_w,
                      spConfig.keypointThreshold );
  removeBorders( mKeypoints, scores_vec, spConfig.border, semi_feature_map_h, semi_feature_map_w );
  topKkeypoints( mKeypoints, scores_vec, spConfig.maxKeypoints );

  features.resize( 259, mKeypoints.size() );
  int desc_feature_dim = mDescDims.d[1];
  int desc_feature_map_h = mDescDims.d[2];
  int desc_feature_map_w = mDescDims.d[3];
  sampleDescriptors( mKeypoints, output_desc, mDescriptors, desc_feature_dim, desc_feature_map_h,
                     desc_feature_map_w );

  for ( int i = 0; i < scores_vec.size(); i++ ) {
    features( 0, i ) = scores_vec[i];
  }

  for ( int i = 1; i < 3; ++i ) {
    for ( int j = 0; j < mKeypoints.size(); ++j ) {
      features( i, j ) = mKeypoints[j][i - 1];
    }
  }

  for ( int m = 3; m < 259; ++m ) {
    for ( int n = 0; n < mDescriptors.size(); ++n ) {
      features( m, n ) = mDescriptors[n][m - 3];
    }
  }

  return true;
}

bool SuperPoint::infer( const cv::Mat &image,
                        Eigen::Matrix<double, 259, Eigen::Dynamic> &features ) {
  auto mContext =
      trtCommon::trtUniquePtr<nvinfer1::IExecutionContext>( mEngine->createExecutionContext() );
  if ( !mContext ) {
    return false;
  }

  assert( mEngine->getNbBindings() == 3 );
  assert( spConfig.inputTensorNames.size() == 1 );

  const int input_index = mEngine->getBindingIndex( spConfig.inputTensorNames[0].c_str() );
  mContext->setBindingDimensions( input_index, nvinfer1::Dims4( 1, 1, image.rows, image.cols ) );
  trtCommon::BufferManager buffers( mEngine, 0, mContext.get() );
  if ( !processInput( buffers, image ) ) {
    return false;
  }

  buffers.copyInputToDevice();

  bool status = mContext->executeV2( buffers.getDeviceBindings().data() );
  if ( !status ) {
    return false;
  }
  buffers.copyOutputToHost();
  if ( !verifyOutput( buffers, features ) ) {
    return false;
  }
  return true;
}

void SuperPoint::saveResultToDisk( const std::string &save_path,
                                   Eigen::Matrix<double, 259, Eigen::Dynamic> &features ) {
  std::ofstream fout( save_path );
  for ( int i = 0; i < features.cols(); i++ ) {
    for ( int j = 0; j < 259; j++ ) {
      fout << features.col( i )( j ) << " ";
    }
    fout << std::endl;
  }
}

}  // namespace dfm