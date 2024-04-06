
add_library(feature_extraction
        orb/orb_feature.cc
        harris/harris.cc
        orb_slam/orbpattern.cpp
        orb_slam/orbextractor.cpp
)
add_library(feature_tracking
        optical_flow/lkpyramid.cc
)

add_executable(harris_test example/harris_test.cc)
target_link_libraries(harris_test feature_extraction ${THIRD_PARTY_LIBS} ${OpenCV_LIBS})
add_executable(orb_test example/orb_test.cc)
target_link_libraries(orb_test feature_extraction ${THIRD_PARTY_LIBS} ${OpenCV_LIBS})
add_executable(orb_slam_feat_test example/orbslam_feat_test.cc)
target_link_libraries(orb_slam_feat_test feature_extraction feature_tracking ${THIRD_PARTY_LIBS} ${OpenCV_LIBS})

add_executable(lk_optical_flow_test example/lk_optical_flow_test.cc)
target_link_libraries(lk_optical_flow_test feature_extraction feature_tracking ${THIRD_PARTY_LIBS} ${OpenCV_LIBS})