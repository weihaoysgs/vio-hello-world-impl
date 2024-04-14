#include "vio_hw/internal/feat/superpoint_impl.hpp"

#include <glog/logging.h>

#ifdef ENABLE_DFM

viohw::SuperPointImpl::SuperPointImpl( const SuperPointExtractorConfig &config ) {
  dfm::readSpLgParameter( config.config_file_path_, sp_config_, lg_config_ );
  sp_config_.maxKeypoints = config.max_kps_;
  tobe_extractor_kps_num_ = config.max_kps_;
  superpoint_infer_.reset( new dfm::SuperPoint( sp_config_ ) );
  if ( !superpoint_infer_->build() ) {
    LOG( FATAL ) << "SuperPoint build failed";
  }
  cv::Mat img( lg_config_.imageHeight, lg_config_.imageWidth, CV_8UC1 );
  cv::randu( img, cv::Scalar( 0 ), cv::Scalar( 255 ) );
  Eigen::Matrix<double, 259, Eigen::Dynamic> feat;
  if ( !superpoint_infer_->infer( img, feat ) ) {
    LOG( FATAL ) << "SuperPoint infer failed";
  }
  LOG( INFO ) << "Superpoint Build Success";
}

bool viohw::SuperPointImpl::detect( const cv::Mat &image, std::vector<cv::KeyPoint> &kps,
                                    cv::Mat &mask, cv::Mat &desc,
                                    Eigen::Matrix<double, 259, Eigen::Dynamic> &feat ) {
  LOG_ASSERT( image.channels() == 1 );
  superpoint_infer_->setMaxExtractorKpsNumber( tobe_extractor_kps_num_ );

  superpoint_infer_->infer( image, feat );
  for ( size_t i = 0; i < feat.cols(); i++ ) {
    double s = feat.col( i )( 0 );
    double x = feat.col( i )( 1 );
    double y = feat.col( i )( 2 );
    kps.emplace_back( cv::Point2f( x, y ), s );
  }
  /*
  cv::Mat draw_img;
  cv::cvtColor( image, draw_img, cv::COLOR_GRAY2BGR );
  for (const auto & kp : kps) {
    cv::Point2i pt( kp.pt );
    cv::circle( draw_img, pt, 1, cv::Scalar( 0, 255, 0 ), -1 );
  }
  cv::imshow("sp", draw_img);
  cv::waitKey(1);
  */
  return true;
}

#endif
