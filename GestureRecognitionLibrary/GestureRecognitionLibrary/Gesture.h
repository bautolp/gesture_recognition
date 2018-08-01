/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

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
	GestureInfo()
	{
		thumb = In;
		fingers = Invalid;
	}
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
	cv::Point& GetHighPoint();

	void SetTime(std::chrono::steady_clock::time_point time);
	void SetHighPoint(cv::Point& hi_pt);
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
		cv::Point hi_pt;
		GestureInfoEx()
		{
			gest_info.fingers = Invalid;
			gest_info.thumb = In;
			gest_info.side = Left; // ?
			centroid.x = 0;
			centroid.y = 0;
			hi_pt.x = 0;
			hi_pt.y = 0;
		}
	};
	GestureInfoEx gesture;
};
