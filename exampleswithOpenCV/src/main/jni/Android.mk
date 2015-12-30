LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include ../../../sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := foe_sample
LOCAL_SRC_FILES := jni_part.cpp CameraCalibration.cpp ARPipeline.cpp GeometryTypes.cpp Pattern.cpp PatternDetector.cpp 
LOCAL_LDLIBS +=  -llog -ldl -lGLESv2

include $(BUILD_SHARED_LIBRARY)
