#include "feat/orb/orb_interface.h"

namespace feat {

class ORB_Impl : public ORB
{
 public:
  explicit ORB_Impl(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
                    int _firstLevel, int _WTA_K, ORB::ScoreType _scoreType, int _patchSize,
                    int _fastThreshold)
      : nfeatures(_nfeatures),
        scaleFactor(_scaleFactor),
        nlevels(_nlevels),
        edgeThreshold(_edgeThreshold),
        firstLevel(_firstLevel),
        wta_k(_WTA_K),
        scoreType(_scoreType),
        patchSize(_patchSize),
        fastThreshold(_fastThreshold)
  {
  }

  void setMaxFeatures(int maxFeatures) CV_OVERRIDE { nfeatures = maxFeatures; }
  int getMaxFeatures() const CV_OVERRIDE { return nfeatures; }

  void setScaleFactor(double scaleFactor_) CV_OVERRIDE { scaleFactor = scaleFactor_; }
  double getScaleFactor() const CV_OVERRIDE { return scaleFactor; }

  void setNLevels(int nlevels_) CV_OVERRIDE { nlevels = nlevels_; }
  int getNLevels() const CV_OVERRIDE { return nlevels; }

  void setEdgeThreshold(int edgeThreshold_) CV_OVERRIDE { edgeThreshold = edgeThreshold_; }
  int getEdgeThreshold() const CV_OVERRIDE { return edgeThreshold; }

  void setFirstLevel(int firstLevel_) CV_OVERRIDE
  {
    CV_Assert(firstLevel_ >= 0);
    firstLevel = firstLevel_;
  }
  int getFirstLevel() const CV_OVERRIDE { return firstLevel; }

  void setWTA_K(int wta_k_) CV_OVERRIDE { wta_k = wta_k_; }
  int getWTA_K() const CV_OVERRIDE { return wta_k; }

  void setScoreType(ORB::ScoreType scoreType_) CV_OVERRIDE { scoreType = scoreType_; }
  ORB::ScoreType getScoreType() const CV_OVERRIDE { return scoreType; }

  void setPatchSize(int patchSize_) CV_OVERRIDE { patchSize = patchSize_; }
  int getPatchSize() const CV_OVERRIDE { return patchSize; }

  void setFastThreshold(int fastThreshold_) CV_OVERRIDE { fastThreshold = fastThreshold_; }
  int getFastThreshold() const CV_OVERRIDE { return fastThreshold; }

  // returns the descriptor size in bytes
  int descriptorSize() const;
  // returns the descriptor type
  int descriptorType() const;
  // returns the default norm type
  int defaultNorm() const;

  // Compute the ORB_Impl features and descriptors on an image
  void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false);

 protected:
  int nfeatures;
  double scaleFactor;
  int nlevels;
  int edgeThreshold;
  int firstLevel;
  int wta_k;
  ORB::ScoreType scoreType;
  int patchSize;
  int fastThreshold;
};

}  // namespace feat