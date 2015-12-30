#include "ARPipeline.hpp"

ARPipeline::ARPipeline(const CameraCalibration& calibration)
  : m_calibration(calibration)
{
}


ARPipeline::ARPipeline(const cv::Mat& patternImage, const CameraCalibration& calibration)
  : m_calibration(calibration)
{
  m_patternDetector.buildPatternFromImage(patternImage, m_query_pattern);

  m_patternDetector.train(m_query_pattern);
}

bool ARPipeline::processFrame(const cv::Mat& displayFrame, const cv::Mat& inputFrame)
{
  bool patternFound = false;

  if (m_train_pattern.keypoints.empty()) 
  {
	m_patternDetector.findFeatures(inputFrame,m_train_pattern);
  }
  else
  {
  	patternFound = m_patternDetector.findFoE(displayFrame, inputFrame, m_train_pattern, m_query_pattern, m_patternInfo);
  }  

  return patternFound;
}

const Transformation& ARPipeline::getPatternLocation() const
{
  return m_patternInfo.pose3d;
}
