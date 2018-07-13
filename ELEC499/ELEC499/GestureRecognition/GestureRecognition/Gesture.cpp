#include "Gesture.h"
#include "stdafx.h"

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
	string gesture_str = "";
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
