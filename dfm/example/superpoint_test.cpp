#include "dfm/internal/superpoint.hpp"

#include <gflags/gflags.h>

DEFINE_string( config_file_path, "../dfm/params/splg_config.yaml", "config file path" );
DEFINE_string( image_path, "../dfm/images/DSC_0410.JPG", "first image path" );
DEFINE_string( ress_save_path, "../dfm/images/DSC_0411.JPG", "first image path" );

int main( int argc, char** argv ) {
  google::ParseCommandLineFlags( &argc, &argv, true );
  dfm::SuperPointConfig spConfig;
  dfm::LightGlueConfig lgConfig;
  dfm::readSpLgParameter( FLAGS_config_file_path, spConfig, lgConfig );

  cv::Mat image_color = cv::imread( FLAGS_image_path );
  assert( !image_color.empty() );

  cv::Mat image;
  cv::cvtColor( image_color, image, cv::COLOR_BGR2GRAY );

  dfm::SuperPoint superpoint_infer( spConfig );
  if ( !superpoint_infer.build() ) {
    std::cerr << "SuperPoint build failed\n";
    return -1;
  }

  Eigen::Matrix<double, 259, Eigen::Dynamic> feat;
  if ( !superpoint_infer.infer( image, feat ) ) {
    std::cerr << "Superpoint infer failed\n";
    return -1;
  }
  std::vector<std::vector<int>> keypoints = superpoint_infer.getKeypoints();

  for ( int i = 0; i < keypoints.size(); i++ ) {
    cv::Point2i pt( keypoints[i][0], keypoints[i][1] );
    cv::circle( image_color, pt, 1, cv::Scalar( 0, 255, 0 ), -1 );
  }
  cv::imshow( "superpoint image", image_color );
  cv::waitKey( 0 );
  return 0;
}