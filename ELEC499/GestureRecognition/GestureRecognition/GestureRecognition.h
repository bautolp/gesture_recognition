#ifndef GestureRecognition_H
#define GestureRecognition_H
#include "BackgroundImageQueue.h"
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <string>
#include <windows.h>

enum GestureDetected
{
	Invalid,
	Fist,
	PalmClosed,
	FingersExtended,
	PinkyIndexExtended,
	MiddleIndexExtended,
};

struct GestureInfo
{
	GestureDetected gesture;
	cv::Rect rect;
	cv::Point centroid;
	bool thumb_extended;
	GestureInfo()
	{
		gesture = Invalid;
		thumb_extended = false;
		centroid.x = 0;
		centroid.y = 0;
	}
};

class GestureRecognition
{
public:
	GestureRecognition();
	~GestureRecognition();

	std::vector<GestureInfo> GetGestureInfo(cv::cuda::GpuMat image);

private:
	struct ThreadInfo
	{
		GestureRecognition* gest_ptr;
		cv::cuda::GpuMat image;
		std::vector<GestureInfo>* gesture_info;
		volatile bool complete;
	};
	struct SkinInfo
	{
		cv::Ptr<cv::cuda::CascadeClassifier>* face_cascade;
		cv::cuda::GpuMat image;
		cv::cuda::GpuMat* output_image;
	};
	void ContourExamination(std::vector<std::vector<cv::Point>>& contours,
							std::vector<GestureInfo>* gesture_info,
							cv::cuda::GpuMat* image);
	void BackgroundSubtraction(cv::cuda::GpuMat* image);
	bool SkinAnalysis(cv::cuda::GpuMat* image, std::vector<cv::Rect> fists, std::vector<cv::Rect> palms);
	static DWORD WINAPI FindFists(LPVOID lpParam);
	static DWORD WINAPI FindPalms(LPVOID lpParam);
	static DWORD WINAPI SkinExtraction(LPVOID lpParam);
	std::vector<std::vector<cv::Point>> GetContours(cv::cuda::GpuMat* image);
	std::vector<cv::Vec4i> GetDefects(std::vector<cv::Point>& contours);

	struct RunTimeOptions
	{
		uint64_t background_image_count;
		uint64_t image_payload_size;
		bool dynamic_background_subtraction;
	};
	RunTimeOptions m_runtime_options;
	cv::Ptr<cv::cuda::CascadeClassifier> m_face_cascade;
	cv::Ptr<cv::cuda::CascadeClassifier> m_eye_cascade;
};

#endif
