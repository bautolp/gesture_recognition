#ifndef GestureRecognition_H
#define GestureRecognition_H
#include "Gesture.h"
#include "GestureTracker.h"
#include <mutex>
#include <string>
#include <windows.h>

// http://www.wseas.us/e-library/conferences/2011/Mexico/CEMATH/CEMATH-20.pdf
#define COLOR_BLUE CV_RGB(0, 0, 255)
#define COLOR_DARK_BLUE CV_RGB(0, 0, 128)
#define COLOR_RED CV_RGB(255, 0, 0)
#define COLOR_DARK_RED CV_RGB(128, 0, 0)
#define COLOR_GREEN CV_RGB(0, 255, 0)
#define COLOR_DARK_GREEN CV_RGB(0, 128, 0)
#define COLOR_TEAL CV_RGB(0, 255, 255)
#define COLOR_DARK_TEAL CV_RGB(0, 128, 128)
#define COLOR_PURPLE CV_RGB(255, 0, 255)
#define COLOR_DARK_PURPLE CV_RGB(128, 0, 128)
#define COLOR_YELLOW CV_RGB(255, 255, 0)
#define COLOR_DARK_YELLOW CV_RGB(128, 128, 0)
#define COLOR_BROWN CV_RGB(128, 72, 0)
#define COLOR_PINK CV_RGB(242, 157, 206)
#define COLOR_AQUA CV_RGB(37, 117, 85)

class GestureRecognition
{
public:
	GestureRecognition();
	~GestureRecognition();

	std::vector<Gesture> GetGestureInfo(cv::cuda::GpuMat image, std::chrono::steady_clock::time_point timestamp);

private:
	struct FaceInfo
	{
		cv::Mat skin_mask;
		cv::Rect face_rect;
	};
	struct SkinInfo
	{
		cv::Ptr<cv::cuda::CascadeClassifier>* face_cascade;
		std::vector<FaceInfo> face_info;
		cv::cuda::GpuMat image;
		std::mutex* cuda_stream;
	};

	void FingerDetection(std::vector<cv::Point>& contours,
						 Gesture& gesture,
						 std::vector<cv::Vec4i>& defects,
						 cv::Mat& image,
						 cv::Rect& face,
						 cv::Point& hi_point,
						 cv::Point& l_point,
						 cv::Point& r_point);
	bool ContourIsFace(std::vector<cv::Point>& contours, cv::Rect& face, cv::Mat& image);
	void FilterContours(std::vector<std::vector<cv::Point>>& contours, cv::Rect& faces, cv::Mat& image);
	void ContourExamination(std::vector<std::vector<cv::Point>>& contours,
							std::vector<Gesture>* gestures,
							cv::Rect& faces,
							cv::Mat& image);
	void DetermineGesture(Gesture* gesture, int fing_cnt);
	bool AddIfNotInVector(std::vector<cv::Point>* points, cv::Point& point, int face_width);
	static DWORD WINAPI SkinExtraction(LPVOID lpParam);
	std::vector<std::vector<cv::Point>> GetContours(cv::Mat* image);
	std::vector<cv::Vec4i> GetDefects(std::vector<cv::Point>& contours);
	void DisplayPoints(cv::Mat& image,
					   cv::Point& start_pt,
					   cv::Point& far_pt,
					   cv::Point& end_pt,
					   CvScalar& colour_1,
					   CvScalar& colour_2,
					   CvScalar& colour_3);
	void FilterDefects(std::vector<cv::Vec4i>& defects,
					   cv::Rect& face,
					   std::vector<cv::Point>& contours,
					   cv::Point& hi_point);

	std::mutex* m_cuda_stream;
	cv::Ptr<cv::cuda::CascadeClassifier> m_face_cascade;
};

#endif
