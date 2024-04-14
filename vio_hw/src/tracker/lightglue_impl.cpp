#include "vio_hw/internal/tracker/lightglue_impl.hpp"

#include <glog/logging.h>

namespace viohw {

LightGlueImpl::LightGlueImpl( const LightGlueImplConfig &config ) {
#ifdef ENABLE_DFM
  // read parameter
  dfm::readSpLgParameter( config.config_file_path_, sp_config_, lg_config_ );
  sp_config_.maxKeypoints = config.max_kps_num_;
  superpoint_infer_.reset( new dfm::SuperPoint( sp_config_ ) );
  lightglue_infer_.reset( new dfm::LightGlue( lg_config_ ) );

  // build engine
  bool sp_build_status = superpoint_infer_->build();
  bool lg_build_status = lightglue_infer_->build();
  CHECK( sp_build_status && lg_build_status ) << "DFM-Matcher, Sp,Lg build failed";

  // first infer build context
  Eigen::Matrix<double, 259, Eigen::Dynamic> feat0, feat1;
  cv::Mat img( lg_config_.imageHeight, lg_config_.imageWidth, CV_8UC1 );
  cv::randu( img, cv::Scalar( 0 ), cv::Scalar( 255 ) );
  LOG_ASSERT( superpoint_infer_->infer( img, feat0 ) ) << "DFM-Sp Infer Failed";
  feat1.resize( 259, 200 );
  feat1.setRandom();
  std::vector<cv::DMatch> matcher;
  CHECK( lightglue_infer_->matcher( feat0, feat1, matcher, true ) ) << "DFM Matcher Infer Failed";
#else
  LOG( FATAL ) << "NOT Enable DFM Model, LightGlue Is Not Build";
#endif
}

void drawLightGlueMatcherResult( const Eigen::Matrix<double, 259, Eigen::Dynamic> &feat0,
                                 const Eigen::Matrix<double, 259, Eigen::Dynamic> &feat1,
                                 std::vector<std::tuple<int, int, float>> &result, cv::Mat left_img,
                                 cv::Mat right_image ) {
  cv::Mat matcher_image_gray, matcher_image_color, line_image;
  cv::hconcat( left_img, right_image, matcher_image_gray );
  cv::cvtColor( matcher_image_gray, matcher_image_color, cv::COLOR_GRAY2BGR );
  cv::cvtColor( left_img, line_image, cv::COLOR_GRAY2BGR );
  size_t image0_width = left_img.cols;
  for ( int i = 0; i < result.size(); i++ ) {
    size_t index0 = std::get<0>( result[i] );
    size_t index1 = std::get<1>( result[i] );
    int x0 = feat0.col( index0 )( 1 );
    int y0 = feat0.col( index0 )( 2 );
    int x1 = feat1.col( index1 )( 1 );
    int y1 = feat1.col( index1 )( 2 );
    cv::Point2i pt0( x0, y0 );
    cv::Point2i pt1( x1 + image0_width, y1 );
    cv::circle( matcher_image_color, pt0, 1, cv::Scalar( 0, 255, 0 ), -1 );
    cv::circle( matcher_image_color, pt1, 1, cv::Scalar( 0, 255, 0 ), -1 );
    cv::line( matcher_image_color, pt0, pt1, cv::Scalar( 0, 255, 0 ), 1 );
    cv::line( line_image, pt0, cv::Point2i( x1, y1 ), cv::Scalar( 0, 255, 0 ), 1, cv::LINE_AA );
    std::printf( "x1 %d y1 %d x2 %d y2 %d\n", x0, y0, x1, y1 );
  }
  cv::imshow( "matcher", matcher_image_color );
  cv::imshow( "stereo", line_image );
  cv::waitKey( 0 );
}

void LightGlueImpl::trackerAndMatcher( const std::vector<cv::Mat> &vprevpyr,
                                       const std::vector<cv::Mat> &curpyr,
                                       std::vector<cv::Point2f> &vkps,
                                       std::vector<cv::Point2f> &vpriorkps,
                                       std::vector<bool> &vkpstatus,
                                       Eigen::Matrix<double, 259, Eigen::Dynamic> &feat_prev,
                                       Eigen::Matrix<double, 259, Eigen::Dynamic> &output ) {
#ifdef ENABLE_DFM
  output.resize( 259, vkps.size() );
  vkpstatus.resize( vkps.size() );
  std::fill( vkpstatus.begin(), vkpstatus.end(), false );
  CHECK( feat_prev.cols() == vkps.size() ) << ",feat_prev.cols() == vkps.size()";
  std::vector<cv::DMatch> matcher;

  // detect superpoint feature for current image
  const cv::Mat &cur_image = curpyr[0];
  Eigen::Matrix<double, 259, Eigen::Dynamic> feat_cur;
  {
    std::lock_guard<std::mutex> lck(sp_memory_lock_);
    CHECK( superpoint_infer_->infer( cur_image, feat_cur ) ) << "SP Infer Failed";
  }

  // lightglue matcher
  CHECK( lightglue_infer_->matcher( feat_prev, feat_cur, matcher, true ) )
      << "DFM, LG Infer Failed";

  std::vector<std::tuple<int, int, float>> result = lightglue_infer_->getLGMatcherResult();
  // drawLightGlueMatcherResult( feat_prev, feat_cur, result, vprevpyr[0], cur_image );

  // prepare result
  for (auto & i : result) {
    size_t index0 = std::get<0>( i );
    size_t index1 = std::get<1>( i );
    int x1 = feat_cur.col( index1 )( 1 );
    int y1 = feat_cur.col( index1 )( 2 );
    vkpstatus.at( index0 ) = true;
    cv::Point2f cur_kp( x1, y1 );
    vpriorkps.at( index0 ) = cur_kp;
    output.col( index0 ) = feat_cur.col( index1 );
  }
#else
  LOG( FATAL ) << "NOT Enable DFM Model, LightGlue Is Not Build";
#endif
}
}  // namespace viohw
