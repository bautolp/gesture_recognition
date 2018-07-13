#include "GestureTracker.h"
#include <iostream>
#include <thread>

using namespace cv;
using namespace cuda;
using namespace std;
using namespace chrono;

#define LOCK_MUTEX(mux) lock_guard<mutex> lock((mux))

GestureTracker::GestureTracker() : gesture_poll(NULL)
{
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
	for (size_t existing_idx = 0; existing_idx < gesture_queue.size(); existing_idx++)
	{
		// cout << "Time diff: " << GetTimeDifference(gesture_queue[existing_idx]) << endl;
		while (GetTimeDifference(gesture_queue[existing_idx]) > QUEUE_HOLD_TIME_MS)
		{
			gesture_queue[existing_idx].pop_front();
		}
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
	LOCK_MUTEX(queuetex);
	// cout << "Looping " << gesture_queue.size() << " gestures." << endl;
	for (size_t queue_idx = 0; queue_idx < gesture_queue.size(); queue_idx++)
	{
		// Absolutely not thread safe, thank god for mutexes
		time_point<steady_clock> start_time;
		if (gesture_queue[queue_idx].size())
		{
			start_time = gesture_queue[queue_idx][0].GetTime();
		}
		uint32_t successive_count = 0;
		GestureInfo last_gesture;
		last_gesture.fingers = Invalid;
		last_gesture.thumb = In;
		last_gesture.side = Left;
		GestureInfo starting_gesture;
		starting_gesture.fingers = Invalid;
		starting_gesture.thumb = In;
		starting_gesture.side = Left;
		GestureInfo ending_gesture;
		ending_gesture.fingers = Invalid;
		ending_gesture.thumb = In;
		ending_gesture.side = Left;
		std::vector<size_t> indexes_of_starting;
		std::vector<size_t> indexes_to_nuke;
		// cout << "Gesture queue size: " << gesture_queue[queue_idx].size() << endl;
		for (size_t gest_idx = 0; gest_idx < gesture_queue[queue_idx].size(); gest_idx++)
		{
			// I think that the side may not be getting pushed in?
			GestureInfo* gest = gesture_queue[queue_idx][gest_idx].GetGesture();
			time_point<steady_clock> gest_time = gesture_queue[queue_idx][gest_idx].GetTime();
			double curr_time = (duration_cast<duration<double>>(gest_time - start_time).count());
			if ((last_gesture.fingers != gest->fingers) || (last_gesture.thumb != gest->thumb))
			{
				last_gesture.fingers = gest->fingers;
				last_gesture.thumb = gest->thumb;
				start_time = gest_time;
				indexes_of_starting.clear();
				indexes_of_starting.push_back(gest_idx);
			}
			else if ((curr_time * 1000) > POSITION_HOLD_TIME_MS)
			{
				// starting hasn't been set, set it
				if (starting_gesture.fingers == Invalid)
				{
					indexes_of_starting.push_back(gest_idx);
					starting_gesture.fingers = gest->fingers;
					starting_gesture.thumb = gest->thumb;
					for (size_t idx = 0; idx < indexes_of_starting.size(); idx++)
					{
						indexes_to_nuke.push_back(indexes_of_starting[idx]);
					}
				}
				// Starting has been set, need to set ending and break out if it is different than starting
				else if ((starting_gesture.fingers != gest->fingers) || (starting_gesture.thumb != gest->thumb))
				{
					ending_gesture.fingers = gest->fingers;
					ending_gesture.thumb = gest->thumb;
					break;
				}
				else
				{
					// cout << "Found starting again" << endl;
					indexes_to_nuke.push_back(gest_idx);
				}
			}
			else
			{
				successive_count++;
				indexes_of_starting.push_back(gest_idx);
			}
		}
		// Presumably we check this frequently enough that we can now empty the queue of the starting gesture
		if (starting_gesture.fingers != Invalid && ending_gesture.fingers != Invalid)
		{
			if (!indexes_to_nuke.size())
				cout << "THIS WILL FUCK UP!" << endl;

			CallbackParams* callback;
			if (callback = FindCallback(starting_gesture, ending_gesture))
			{
				cout << "Transition found " << starting_gesture.fingers << " fingers and thumb "
					 << (starting_gesture.thumb == In ? "in" : "out") << " to " << ending_gesture.fingers
					 << " fingers and thumb " << (ending_gesture.thumb == In ? "in" : "out") << endl;
				callback_queue.push_back(callback->function_call);
				size_t first_idx = 0xFFFFFFFF;
				size_t last_idx = 0;
				for (size_t idx = 0; idx < indexes_to_nuke.size(); idx++)
				{
					if (first_idx > idx)
						first_idx = idx;
					if (last_idx < idx)
						last_idx = idx;
				}
				// There has got to be a better way to get this info v___v
				cout << "Removing " << first_idx << " to " << last_idx << endl;
				gesture_queue[queue_idx].erase(gesture_queue[queue_idx].begin() + first_idx,
											   gesture_queue[queue_idx].begin() + last_idx);
			}
		}
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
