#pragma once
#include <chrono>
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <stdint.h>
#include <string>
enum ThumbPosition
{
	In,
	Out
};
enum HandSide
{
	Left,
	Right
};
enum FingerCount
{
	Invalid,
	Fist,
	OneFinger,
	TwoFingers,
	ThreeFingers,
	FourFingers
};

struct GestureInfo
{
	ThumbPosition thumb;
	HandSide side;
	FingerCount fingers;
};

class Gesture
{
public:
	Gesture();
	~Gesture();
	std::string GestureToString();
	GestureInfo* GetGesture();
	std::chrono::steady_clock::time_point GetTime();
	cv::Point GetCentroid();
	cv::Rect GetFace();
	HandSide GetSide();
	FingerCount GetFingerCount();
	ThumbPosition GetThumb();

	void SetTime(std::chrono::steady_clock::time_point time);
	void SetCentroid(cv::Point);
	void SetFace(cv::Rect);
	void SetGesture(GestureInfo& gest);
	void SetFingerCount(FingerCount);
	void SetThumb(ThumbPosition thumb);
	void SetSide(HandSide side);

private:
	struct GestureInfoEx
	{
		GestureInfo gest_info;
		cv::Point centroid;
		std::chrono::steady_clock::time_point time_pt;
		cv::Rect face;
		GestureInfoEx()
		{
			gest_info.fingers = Invalid;
			gest_info.thumb = In;
			gest_info.side = Left; // ?
			centroid.x = 0;
			centroid.y = 0;
		}
	};
	GestureInfoEx gesture;
};
