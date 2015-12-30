#ifndef ARPIPELINE_HPP
#define ARPIPELINE_HPP

////////////////////////////////////////////////////////////////////
// File includes:
#include "PatternDetector.hpp"
#include "CameraCalibration.hpp"
#include "GeometryTypes.hpp"

class ARPipeline
{
public:
  ARPipeline(const CameraCalibration& calibration);

  ARPipeline(const cv::Mat& patternImage, const CameraCalibration& calibration);

  bool processFrame(const cv::Mat& displayFrame, const cv::Mat& inputFrame);

  const Transformation& getPatternLocation() const;

  PatternDetector     m_patternDetector;
private:

private:
  CameraCalibration   m_calibration;
  Pattern             m_query_pattern;
  Pattern             m_train_pattern;
  PatternTrackingInfo m_patternInfo;
  //PatternDetector     m_patternDetector;
};

#endif
