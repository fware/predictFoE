#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <set>
#include "CameraCalibration.hpp"
#include "ARPipeline.hpp"

using namespace std;
using namespace cv;

extern "C" {

// Change this calibration to yours:
CameraCalibration calibration(526.58037684199849f, 524.65577209994706f, 318.41744018680112f, 202.96659047014398f);
ARPipeline pipeline(calibration);


//Member methods
/**
 * Performs full detection routine on camera frame and draws the scene using drawing context.
 * In addition, this function draw overlay with debug information on top of the AR window.
 * Returns true if processing loop should be stopped; otherwise - false.
 */
bool processFrame(const cv::Mat& displayFrame, const cv::Mat& cameraFrame, ARPipeline& pipeline);
JNIEXPORT void JNICALL Java_com_wareshopc_example_cv_foe_FocusOfExpansionActivity_Init(JNIEnv*, jobject);
JNIEXPORT void JNICALL Java_com_wareshopc_example_cv_foe_FocusOfExpansionActivity_FindFeatures(JNIEnv*, jobject, jlong fcount, jlong addrGray1, /*jlong addrGray2,*/ jlong addrRgba1 /*2*/);
void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);
void PointsToKeyPoints(const std::vector<cv::Point2f>& ps, std::vector<cv::KeyPoint>& kps);
double euclideanDist(Point2f p, Point2f q);

JNIEXPORT void JNICALL Java_com_wareshopc_example_cv_foe_FocusOfExpansionActivity_Init(JNIEnv*, jobject)
{
    // Change this calibration to yours:
    //CameraCalibration calibration(526.58037684199849f, 524.65577209994706f, 318.41744018680112f, 202.96659047014398f);
	//ARPipeline pipeline(calibration);
}
JNIEXPORT void JNICALL Java_com_wareshopc_example_cv_foe_FocusOfExpansionActivity_FindFeatures(JNIEnv*, jobject, jlong fcount, jlong addrGray1, jlong addrRgba1)
{	
	long frameCount = fcount;
	Mat& mGr1 = *(Mat*) addrGray1;
    Mat& mRgb = *(Mat*)addrRgba1;

	processFrame(mRgb, mGr1, pipeline);
	frameCount++;
}


bool processFrame(const cv::Mat& displayFrame, const cv::Mat& cameraFrame, ARPipeline& pipeline)
{
	pipeline.processFrame(displayFrame, cameraFrame);

    return true; //shouldQuit;
}


}
