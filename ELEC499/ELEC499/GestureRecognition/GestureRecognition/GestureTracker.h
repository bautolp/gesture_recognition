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
	GestureTracker::GestureTracker();
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
