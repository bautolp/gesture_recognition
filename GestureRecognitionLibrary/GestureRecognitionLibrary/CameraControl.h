/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

#pragma once
#include "ArenaApi.h"
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class CameraControl
{
public:
	CameraControl();
	~CameraControl();

	cv::cuda::GpuMat GetMat();
	void SetupCamera(int64_t width, int64_t height);
	void EnumerateCamera();
	void StartAcquisition();
	void StopAcquisition();

private:
	cv::cuda::GpuMat* cv_image;
	cv::cuda::GpuMat* GetCompleteMat();
	Arena::IDevice* m_device;
	GenApi::INodeMap* m_node_map;
	Arena::ISystem* m_system;
};
