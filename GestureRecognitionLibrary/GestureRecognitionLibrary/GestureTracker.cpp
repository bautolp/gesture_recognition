/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

#include "GestureTracker.h"
#include <iomanip> // setprecision
#include <iostream>
#include <sstream> // stringstream
#include <thread>

using namespace cv;
using namespace cuda;
using namespace std;
using namespace chrono;

#define LOCK_MUTEX(mux) lock_guard<mutex> lock((mux))

GestureTracker::GestureTracker(cv::Size im_size) : gesture_poll(NULL)
{
	m_im_size.width = im_size.width / 2;
	m_im_size.height = im_size.height / 2;
	gesture_poll = new thread(GestureTrackerPoll, this);
}

GestureTracker::~GestureTracker()
{
	if (gesture_poll != NULL)
	{
		exit_thread = true;
		gesture_poll->join();
	}
}

void GestureTracker::AddGesture(vector<Gesture>& gestures)
{
	// Go through the new gestures, checking if any seem to belong to an existing gesture user
	LOCK_MUTEX(queuetex);
	for (size_t gest_idx = 0; gest_idx < gestures.size(); gest_idx++)
	{
		bool found = false;
		if (gestures[gest_idx].GetFace().x == 0 || gestures[gest_idx].GetFace().y == 0)
			throw exception("Face is not in gesture");
		for (size_t existing_idx = 0; existing_idx < gesture_queue.size(); existing_idx++)
		{
			if (!SameFace(gestures[gest_idx], gesture_queue[existing_idx].back()))
			{
				continue;
			}
			if (!SameSide(gestures[gest_idx], gesture_queue[existing_idx].back()))
			{
				continue;
			}
			found = true;
			gesture_queue[existing_idx].push_back(gestures[gest_idx]);
		}
		if (!found)
		{
			deque<Gesture> new_gesture_queue;
			gesture_queue.push_back(new_gesture_queue);
			gesture_queue.back().push_back(gestures[gest_idx]);
		}
	}
}

bool GestureTracker::RegisterCallback(GestureInfo& start_gesture,
									  GestureInfo& end_gesture,
									  std::function<void(void*)> function_call)
{
	if (end_gesture.fingers == Invalid || start_gesture.fingers == Invalid)
		return NULL;
	CallbackParams new_callback;
	new_callback.start.fingers = start_gesture.fingers;
	new_callback.start.thumb = start_gesture.thumb;
	new_callback.end.fingers = end_gesture.fingers;
	new_callback.end.thumb = end_gesture.thumb;
	new_callback.function_call = function_call;
	// Dont need to lock cause its already locked
	CallbackParams* callback = FindCallback(start_gesture, end_gesture);

	// If it is not register, register it and return true, otherwise return false
	if (callback == NULL)
	{
		callback_registry.push_back(new_callback);
		return true;
	}
	return false;
};

void GestureTracker::ReSizeQueue()
{
	// Maybe use current time instead of that? It's more relevant
	LOCK_MUTEX(queuetex);
	try
	{
		for (size_t existing_idx = 0; existing_idx < gesture_queue.size(); existing_idx++)
		{
			// cout << "Time diff: " << GetTimeDifference(gesture_queue[existing_idx]) << endl;
			if (gesture_queue[existing_idx].size() > 1)
			{
				while (GetTimeDifference(gesture_queue[existing_idx]) > QUEUE_HOLD_TIME_MS)
				{
					cout << "Removing olds" << endl;
					gesture_queue[existing_idx].pop_front();
				}
				duration<double> time_span = duration_cast<duration<double>>(
					steady_clock::now() - gesture_queue[existing_idx].front().GetTime());
				if (time_span.count() > 2)
				{
					cout << "Removing old" << endl;
					gesture_queue[existing_idx].pop_front();
				}
			}
		}
	}
	catch (exception& e)
	{
		cout << e.what();
	}
}

bool GestureTracker::SameFace(Gesture& gest_1, Gesture& gest_2)
{
	Rect face_1 = gest_1.GetFace();
	Rect face_2 = gest_2.GetFace();
	Rect intersection = face_1 & face_2;
	return intersection.area() > 0;
}

bool GestureTracker::SameSide(Gesture& gest_1, Gesture& gest_2)
{
	return gest_1.GetSide() == gest_2.GetSide();
}

// Have to prefix type with class name because otherwise the function definition isn't recognized because the class
// scope doesn't take effect until inside the function
GestureTracker::CallbackParams* GestureTracker::FindCallback(GestureInfo& start_gesture, GestureInfo& end_gesture)
{
	if (start_gesture.fingers == Invalid || end_gesture.fingers == Invalid)
		return NULL;
	// Side is not used here!
	LOCK_MUTEX(callback_mutex);
	for (size_t cb_idx = 0; cb_idx < callback_registry.size(); cb_idx++)
	{
		if ((callback_registry[cb_idx].start.fingers == start_gesture.fingers &&
			 callback_registry[cb_idx].start.thumb == start_gesture.thumb) &&
			(callback_registry[cb_idx].end.fingers == end_gesture.fingers &&
			 callback_registry[cb_idx].end.thumb == end_gesture.thumb))
			return &(callback_registry[cb_idx]);
	}
	return NULL;
}

uint32_t GestureTracker::GetTimeDifference(std::deque<Gesture>& gestures)
{
	duration<double> time_span =
		duration_cast<duration<double>>(gestures.back().GetTime() - gestures.front().GetTime());
	return uint32_t(time_span.count() * 1000);
}

void GestureTracker::DetectTransitions()
{
	Mat test(m_im_size, CV_8U, Scalar(0));
	int count = 0;
	namedWindow("Test", CV_WINDOW_NORMAL);
	LOCK_MUTEX(queuetex);
	for (size_t queue_idx = 0; queue_idx < gesture_queue.size(); queue_idx++)
	{
		// Absolutely not thread safe, thank god for mutexes
		time_point<steady_clock> start_time;
		while (gesture_queue[queue_idx].size() > 3)
		{
			count = 3;
			gesture_queue[queue_idx].pop_front();
		}
		uint32_t successive_count = 0;
		GestureInfo last_gesture;
		last_gesture.fingers = Invalid;
		last_gesture.thumb = In;
		last_gesture.side = Left;
		std::vector<size_t> indexes_of_starting;
		std::vector<size_t> indexes_to_nuke;
		bool counting_invalid = false;
		for (size_t gest_idx = 0; gest_idx < gesture_queue[queue_idx].size(); gest_idx++)
		{
			if (!counting_invalid)
			{
				if (gesture_queue[queue_idx][gest_idx].GetGesture()->fingers != Invalid)
				{
					if (last_gesture.fingers != Invalid)
					{
						if (gesture_queue[queue_idx][gest_idx].GetGesture()->fingers == last_gesture.fingers &&
							gesture_queue[queue_idx][gest_idx].GetGesture()->thumb == last_gesture.thumb)
						{
							successive_count++;
						}
					}
					else
					{
						successive_count = 1;
						last_gesture.fingers = gesture_queue[queue_idx][gest_idx].GetGesture()->fingers;
						last_gesture.thumb = gesture_queue[queue_idx][gest_idx].GetGesture()->thumb;
					}
				}
				else
				{
					counting_invalid = true;
					successive_count = 1;
				}
			}
			else
			{
				if (gesture_queue[queue_idx][gest_idx].GetGesture()->fingers == Invalid)
				{
					successive_count++;
				}
			}
		}
		if (gesture_queue[queue_idx].size() == 3)
		{
			if (successive_count == 3)
			{
				if (!counting_invalid)
				{
					int finger_cnt = 0;
					switch (last_gesture.fingers)
					{
						case OneFinger:
							finger_cnt = 1;
							break;
						case TwoFingers:
							finger_cnt = 2;
							break;
						case ThreeFingers:
							finger_cnt = 3;
							break;
						case FourFingers:
							finger_cnt = 4;
							break;
					}
					if (last_gesture.thumb == Out)
					{
						finger_cnt++;
					}
					string out_txt;
					out_txt = to_string(finger_cnt);
					duration<double> time_span = duration_cast<duration<double>>(gesture_queue[queue_idx][2].GetTime() -
																				 gesture_queue[queue_idx][1].GetTime());
					double fps = 1 / time_span.count();
					stringstream stream;
					stream << setprecision(2) << fps;
					string s = stream.str();
					putText(test, out_txt, gesture_queue[queue_idx][2].GetHighPoint(), 5, 5, Scalar(255), 2);
					putText(test, s, Point(0, test.size().height - 10), CV_FONT_ITALIC, 1, Scalar(255), 1);
					imshow("Test", test);
					waitKey(10);
				}
				else
				{
					imshow("Test", test);
					waitKey(10);
				}
			}
		}
	}
	if (count == 0)
	{
		imshow("Test", test);
		waitKey(10);
	}
}

void GestureTracker::CallCallbackQueue()
{
	for (size_t idx = 0; idx < callback_queue.size(); idx++)
	{
		cout << "Calling callback" << endl;
		callback_queue[idx](NULL);
		// thread callback(&CallFunction, callback_queue[idx]);
	}
	if (callback_queue.size())
	{
		callback_queue.clear();
	}
}

void GestureTracker::GestureTrackerPoll(LPVOID gesture_tracker_ptr)
{
	GestureTracker* gesture_tracker = (GestureTracker*)gesture_tracker_ptr;
	while (!gesture_tracker->exit_thread)
	{
		Sleep(100);
		gesture_tracker->ReSizeQueue();
		gesture_tracker->DetectTransitions();
		gesture_tracker->CallCallbackQueue();
	}
}
