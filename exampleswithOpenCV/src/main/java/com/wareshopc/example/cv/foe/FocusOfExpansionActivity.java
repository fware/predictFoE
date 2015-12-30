package com.wareshopc.example.cv.foe;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import com.wareshopc.example.cv.foe.R;
import com.wareshopc.example.cv.foe.R.id;
import com.wareshopc.example.cv.foe.R.layout;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class FocusOfExpansionActivity extends Activity implements CvCameraViewListener2 {
    private static final String    TAG = "OCVSample::Activity";

    private static final int       VIEW_MODE_RGBA     = 0;
    private static final int       VIEW_MODE_GRAY     = 1;
    private static final int       VIEW_MODE_CANNY    = 2;
    private static final int       VIEW_MODE_FEATURES = 5;

    private int                    mViewMode;
    private Mat                    mRgba1;
    private Mat                    mIntermediateMat;
    private Mat                    mGray1;
    
    //members for optical flow framework
    private long mFrameCount;
    private Mat mInputFrame1;
    private Mat mInputFrame2;
    private Mat mInputFrame1Gray;
    private Mat mInputFrame2Gray;
    private boolean mFrameToggle;

    private MenuItem               mItemPreviewRGBA;
    private MenuItem               mItemPreviewGray;
    private MenuItem               mItemPreviewCanny;
    private MenuItem               mItemPreviewFeatures;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("foe_sample");

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FocusOfExpansionActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //Intializing variables for algorithm execution
        mFrameToggle = false;
        mFrameCount = 0;

        
        setContentView(R.layout.predictfoe_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.predictfoe_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewGray = menu.add("Preview GRAY");
        mItemPreviewCanny = menu.add("Canny");
        mItemPreviewFeatures = menu.add("Find features");
        return true;
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba1 = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray1 = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba1.release();
        mGray1.release();
    	
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    	
        final int viewMode = VIEW_MODE_FEATURES; //mViewMode;

        if(!mFrameToggle) {
        	mInputFrame1 = inputFrame.rgba();
        	mInputFrame1Gray = inputFrame.gray();        	
    		mFrameToggle = true;
    	}
    	
    	if (mFrameToggle) {
        	mInputFrame1 = inputFrame.rgba();
        	mInputFrame1Gray = inputFrame.gray();    		
    		    		
	        switch (viewMode) {
	        case VIEW_MODE_GRAY:
	            // input frame has gray scale format
	            Imgproc.cvtColor(inputFrame.gray(), mRgba1 /*2*/, Imgproc.COLOR_GRAY2RGBA, 4);
	            break;
	        case VIEW_MODE_RGBA:
	            // input frame has RBGA format
	            mRgba1 = inputFrame.rgba();
	            break;
	        case VIEW_MODE_CANNY:
	            // input frame has gray scale format
	            mRgba1 = inputFrame.rgba();
	            Imgproc.Canny(inputFrame.gray(), mIntermediateMat, 80, 100);
	            Imgproc.cvtColor(mIntermediateMat, mRgba1, Imgproc.COLOR_GRAY2RGBA, 4);
	            break;
	        case VIEW_MODE_FEATURES:
	            mRgba1 = mInputFrame1;
	            mGray1 = mInputFrame1Gray;
	            FindFeatures(mFrameCount, mGray1.getNativeObjAddr(), mRgba1.getNativeObjAddr());
	            mFrameCount++;
	        	
	            break;
	        }
        
    	}
    	
        return mInputFrame1;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemPreviewRGBA) {
            mViewMode = VIEW_MODE_RGBA;
        } else if (item == mItemPreviewGray) {
            mViewMode = VIEW_MODE_GRAY;
        } else if (item == mItemPreviewCanny) {
            mViewMode = VIEW_MODE_CANNY;
        } else if (item == mItemPreviewFeatures) {
            mViewMode = VIEW_MODE_FEATURES;
        }

        return true;
    }

    public native void FindFeatures(long mFrameCount, long matAddrGr1, long matAddrRgba1);
}
