/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

#include "Gesture.h"

using namespace std;
using namespace chrono;
using namespace cv;

Gesture::Gesture()
{
}

Gesture::~Gesture()
{
}

string Gesture::GestureToString()
{
	string gesture_str = (gesture.gest_info.side == Left) ? "Left - " : "Right - ";
	switch (gesture.gest_info.fingers)
	{
		case Invalid:
			gesture_str += "Invalid";
			break;
		case Fist:
			gesture_str += "Fist";
			break;
		case OneFinger:
			gesture_str += "OneFinger";
			break;
		case TwoFingers:
			gesture_str += "TwoFingers";
			break;
		case ThreeFingers:
			gesture_str += "ThreeFingers";
			break;
		case FourFingers:
			gesture_str += "FourFingers";
			break;
	}
	gesture_str += " ";
	if (gesture.gest_info.thumb == Out)
	{
		gesture_str += "with thumb";
	}
	return gesture_str;
}

GestureInfo* Gesture::GetGesture()
{
	return &gesture.gest_info;
}

steady_clock::time_point Gesture::GetTime()
{
	return gesture.time_pt;
}

Point Gesture::GetCentroid()
{
	return gesture.centroid;
}

Rect Gesture::GetFace()
{
	return gesture.face;
}

HandSide Gesture::GetSide()
{
	return gesture.gest_info.side;
}

FingerCount Gesture::GetFingerCount()
{
	return gesture.gest_info.fingers;
}

ThumbPosition Gesture::GetThumb()
{
	return gesture.gest_info.thumb;
}

cv::Point& Gesture::GetHighPoint()
{
	return gesture.hi_pt;
}

void Gesture::SetGesture(GestureInfo& gest)
{
	gesture.gest_info.fingers = gest.fingers;
	gesture.gest_info.side = gest.side;
	gesture.gest_info.thumb = gest.thumb;
}

void Gesture::SetFingerCount(FingerCount finger_count)
{
	gesture.gest_info.fingers = finger_count;
}

void Gesture::SetThumb(ThumbPosition thumb)
{
	gesture.gest_info.thumb = thumb;
}

void Gesture::SetSide(HandSide side)
{
	gesture.gest_info.side = side;
}

void Gesture::SetTime(steady_clock::time_point time)
{
	gesture.time_pt = time;
}

void Gesture::SetCentroid(Point cent)
{
	gesture.centroid = cent;
}

void Gesture::SetFace(Rect face)
{
	gesture.face = face;
}

void Gesture::SetHighPoint(Point& hi_pt)
{
	gesture.hi_pt.x = hi_pt.x;
	gesture.hi_pt.y = hi_pt.y;
}
