#include "dfm/internal/lightglue.hpp"

#include <gflags/gflags.h>

DEFINE_string( config_file_path, "../dfm/params/splg_config.yaml", "config file path" );
DEFINE_string( image0_path, "../dfm/images/DSC_0410.JPG", "first image path" );
DEFINE_string( image1_path, "../dfm/images/DSC_0411.JPG", "first image path" );
DEFINE_string( res_save_path, "../dfm/images/DSC_result.JPG", "first image path" );

int main( int argc, char** argv ) {
  google::ParseCommandLineFlags( &argc, &argv, true );
  std::string config_file_path = FLAGS_config_file_path;
  std::string image0_path = FLAGS_image0_path;
  std::string image1_path = FLAGS_image1_path;
  std::string match_image_save_path = FLAGS_res_save_path;
  cv::Mat image0_color = cv::imread( image0_path );
  cv::Mat image1_color = cv::imread( image1_path );

  dfm::LightGlueConfig lgConfig;
  dfm::SuperPointConfig spConfig;
  dfm::readSpLgParameter( config_file_path, spConfig, lgConfig );
  size_t imageWidth = lgConfig.imageWidth;
  size_t imageHeight = lgConfig.imageHeight;

  cv::resize( image0_color, image0_color, cv::Size( imageWidth, imageHeight ) );
  cv::resize( image1_color, image1_color, cv::Size( imageWidth, imageHeight ) );
  assert( !image0_color.empty() && !image1_color.empty() );
  assert( image0_color.size().width == image1_color.size().width );

  cv::Mat image0, image1;
  cv::cvtColor( image0_color, image0, cv::COLOR_BGR2GRAY );
  cv::cvtColor( image1_color, image1, cv::COLOR_BGR2GRAY );

  dfm::LightGlue lightglue( lgConfig );
  dfm::SuperPoint superpoint( spConfig );

  if ( !lightglue.build() ) {
    std::cerr << "LightGlue build failed\n";
    return -1;
  }

  if ( !superpoint.build() ) {
    std::cerr << "SuperPoint build failed\n";
    return -1;
  }

  Eigen::Matrix<double, 259, Eigen::Dynamic> feat0, feat1;
  superpoint.infer( image0, feat0 );
  superpoint.infer( image1, feat1 );

  std::vector<cv::DMatch> matcher_result;
  bool status = lightglue.matcher( feat0, feat1, matcher_result, true );
  lightglue.matcher( feat0, feat1, matcher_result, true );
  lightglue.matcher( feat0, feat1, matcher_result, true );
  lightglue.matcher( feat0, feat1, matcher_result, true );
  if ( status ) {
    std::cout << "LightGlue Infer success\n";
    std::vector<std::tuple<int, int, float>> result = lightglue.getLGMatcherResult();
    std::cout << "matcher success size: " << result.size() << std::endl;
    cv::Mat matcher_image;
    cv::hconcat( image0_color, image1_color, matcher_image );
    size_t image0_width = image0_color.cols;
    for ( int i = 0; i < result.size(); i++ ) {
      size_t index0 = std::get<0>( result[i] );
      size_t index1 = std::get<1>( result[i] );
      int x0 = feat0.col( index0 )( 1 );
      int y0 = feat0.col( index0 )( 2 );

      int x1 = feat1.col( index1 )( 1 );
      int y1 = feat1.col( index1 )( 2 );

      cv::Point2i pt0( x0, y0 );
      cv::Point2i pt1( x1 + image0_width, y1 );
      cv::circle( matcher_image, pt0, 3, cv::Scalar( 0, 255, 0 ), -1 );
      cv::circle( matcher_image, pt1, 3, cv::Scalar( 0, 255, 0 ), -1 );
      cv::line( matcher_image, pt0, pt1, cv::Scalar( 0, 255, 0 ), 1 );
    }

    cv::imshow( "matcher", matcher_image );
    // cv::imwrite( match_image_save_path, matcher_image );
    cv::waitKey( 0 );
  }

  return 0;
}
