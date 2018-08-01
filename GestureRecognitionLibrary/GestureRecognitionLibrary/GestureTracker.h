/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

#ifndef GestureTracker_H
#define GestureTracker_H

#include "Gesture.h"
#include <Windows.h>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <stdint.h>

#define QUEUE_HOLD_TIME_MS 5000
#define POSITION_HOLD_TIME_MS 200
#define MINIMUM_IN_A_ROW 5
#define FORGIVENESS_FACTOR 1
class GestureTracker
{
public:
	GestureTracker::GestureTracker(cv::Size im_size);
	GestureTracker::~GestureTracker();

	void AddGesture(std::vector<Gesture>& gestures);
	bool RegisterCallback(GestureInfo& start_gesture,
						  GestureInfo& end_gesture,
						  std::function<void(void*)> function_call);

private:
	void GestureTracker::CallFunction(void(*function()))
	{
		function();
	}
	struct CallbackParams
	{
		std::function<void(void*)> function_call;
		GestureInfo start;
		GestureInfo end;
		CallbackParams()
		{
			function_call = NULL;
			start.fingers = Invalid;
			end.fingers = Invalid;
			start.thumb = In;
			end.thumb = In;
			start.side = Left;
			end.side = Left;
		}
	};
	cv::Size m_im_size;
	// TODO prefix with m_
	// A vector of queues, each queue is a different hand
	std::vector<std::deque<Gesture>> gesture_queue; // back newest, front oldest, push goes to back, pop removed front
	std::vector<CallbackParams> callback_registry;
	std::vector<std::function<void(void*)>> callback_queue;
	std::mutex queuetex;
	std::mutex callback_mutex;
	volatile bool exit_thread;
	std::thread* gesture_poll;

	void ReSizeQueue();
	bool SameFace(Gesture& gest_1, Gesture& gest_2);
	bool SameSide(Gesture& gest_1, Gesture& gest_2);
	CallbackParams* GestureTracker::FindCallback(GestureInfo& start_gesture, GestureInfo& end_gesture);
	void CallCallbackQueue();
	uint32_t GetTimeDifference(std::deque<Gesture>& gestures);
	void DetectTransitions();
	static void GestureTrackerPoll(LPVOID gesture_tracker_ptr);
};

#endif
