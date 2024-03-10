
add_library(feature_extraction
        orb/orb_feature.cc
        harris/harris.cc
)

add_executable(harris_test example/harris_test.cc)
target_link_libraries(harris_test feature_extraction ${THIRD_PARTY_LIBS} ${OpenCV_LIBS})