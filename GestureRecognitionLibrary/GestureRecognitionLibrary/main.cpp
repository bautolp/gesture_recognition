/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

#include "CameraControl.h"
#include "Gesture.h"
#include "GestureRecognition.h"
#include "GestureTracker.h"
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;
using namespace cuda;
using namespace chrono;

void callback_a(void* lpParam)
{
	cout << endl << endl << endl << endl;
	cout << "Callback A";
	cout << endl << endl << endl << endl;
}

void RegisterCallbacks(GestureTracker& tracker)
{
	GestureInfo start;
	start.fingers = FourFingers;
	start.thumb = Out;
	GestureInfo end;
	end.fingers = FourFingers;
	end.thumb = Out;
	tracker.RegisterCallback(start, end, callback_a);
}

int main()
{
	try
	{
		CameraControl camera;
		GestureRecognition recognition;
		GestureTracker tracker(Size(1440, 1080));
		camera.EnumerateCamera();
		camera.SetupCamera(1440, 1080);
		camera.StartAcquisition();
		while (1)
		{
			GpuMat image = camera.GetMat();
			vector<Gesture> gestures = recognition.GetGestureInfo(image, steady_clock::now());
			tracker.AddGesture(gestures);
		}
	}
	catch (exception& e)
	{
		cout << "Exception " << e.what();
	}
	cout << "Exiting." << endl;
}
