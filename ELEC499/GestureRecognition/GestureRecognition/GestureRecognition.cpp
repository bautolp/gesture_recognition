#include "GestureExceptions.h"
#include "GestureRecognition.h"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/background_segm.hpp"
#include <Windows.h>
#include <chrono>
#include <iostream>
#include <stdint.h>

// cvtColor(*image, hsv, CV_BGR2HSV);
/* inRange(hsv, Scalar(0, 10, 60), Scalar(20, 150, 255), gray);
// works better at school
inRange(hsv, Scalar(8, 48, 120), Scalar(20, 150, 255), skin_mask);
namedWindow("hsv", 1);
imshow("hsv", skin_mask);
waitKey(0);*/

// Maybe make runtime options and load the class with them
// Could allow cmd line adjustment of runtime options dynamically
// for performance tweaking
#define BACKGROUND_QUEUE_SIZE 100
#define BLUR_SIZE Size(8, 8)
#define MINIMUM_CONTOUR_AREA 5000.0
#define MINIMUM_DEFECT_DEPTH 15.0
#define ANGLE_BETWEEN_FINGERS 55

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

// Worked in lab at school
/*#define LAB_RANGE
#ifdef LAB_RANGE
#define Y_MIN 91
#define Y_MAX 255
#define Cb_MIN 80
#define Cb_MAX 125
#define Cr_MIN 143
#define Cr_MAX 173
#endif*/

#define MIN_CHANCE_95 (unsigned char)(((float)95 / (float)100) * (float)255)
#define MIN_CHANCE_90 (unsigned char)(((float)90 / (float)100) * (float)255)
#define MIN_CHANCE_85 (unsigned char)(((float)85 / (float)100) * (float)255)
#define MIN_CHANCE_80 (unsigned char)(((float)80 / (float)100) * (float)255)
#define MIN_CHANCE_75 (unsigned char)(((float)75 / (float)100) * (float)255)
#define MIN_CHANCE_70 (unsigned char)(((float)70 / (float)100) * (float)255)
#define MIN_CHANCE_65 (unsigned char)(((float)65 / (float)100) * (float)255)
#define MIN_CHANCE_60 (unsigned char)(((float)60 / (float)100) * (float)255)
#define MIN_CHANCE_55 (unsigned char)(((float)55 / (float)100) * (float)255)
#define MIN_CHANCE_50 (unsigned char)(((float)50 / (float)100) * (float)255)
#define MIN_CHANCE_45 (unsigned char)(((float)45 / (float)100) * (float)255)
#define MIN_CHANCE_40 (unsigned char)(((float)40 / (float)100) * (float)255)
#define MIN_CHANCE_35 (unsigned char)(((float)35 / (float)100) * (float)255)
#define MIN_CHANCE_30 (unsigned char)(((float)30 / (float)100) * (float)255)
#define MIN_CHANCE_25 (unsigned char)(((float)25 / (float)100) * (float)255)
#define MIN_CHANCE_20 (unsigned char)(((float)20 / (float)100) * (float)255)
#define MIN_CHANCE_15 (unsigned char)(((float)15 / (float)100) * (float)255)
#define MIN_CHANCE_10 (unsigned char)(((float)10 / (float)100) * (float)255)
#define MIN_CHANCE_5 (unsigned char)(((float)5 / (float)100) * (float)255)
#define MIN_CHANCE_0 0

int min_probability = MIN_CHANCE_50;

#ifndef LAB_RANGE
#define RANGE_1_Y_MIN 115
#define RANGE_1_Y_MAX 200
#define RANGE_1_Cr_MIN 143
#define RANGE_1_Cr_MAX 173
#define RANGE_1_Cb_MIN 80
#define RANGE_1_Cb_MAX 120

#define RANGE_4_Y_MIN 115
#define RANGE_4_Y_MAX 125
#define RANGE_4_Cr_MIN 132
#define RANGE_4_Cr_MAX 142
#define RANGE_4_Cb_MIN 145
#define RANGE_4_Cb_MAX 175

#define RANGE_3_Y_MIN 115
#define RANGE_3_Y_MAX 130
#define RANGE_3_Cr_MIN 135
#define RANGE_3_Cr_MAX 140
#define RANGE_3_Cb_MIN 165
#define RANGE_3_Cb_MAX 175

#define RANGE_2_Y_MIN 115
#define RANGE_2_Y_MAX 130
#define RANGE_2_Cr_MIN 140
#define RANGE_2_Cr_MAX 150
#define RANGE_2_Cb_MIN 90
#define RANGE_2_Cb_MAX 110
#endif

#ifndef pi
#define pi 3.14159265359
#endif

#define EYE_CASCADE_PATH                                                                                               \
	"C:\\Users\\lyndo\\Developer\\OpenCV_"                                                                             \
	"cuda\\install\\etc\\haarcascades\\haarcascade_eye.xml"

#define FACE_CASCADE_PATH                                                                                              \
	"C:\\Users\\lyndo\\Developer\\OpenCV_"                                                                             \
	"cuda\\install\\etc\\haarcascades\\haarcascade_"                                                                   \
	"frontalface_alt.xml"

using namespace cv;
using namespace cuda;
using namespace std;

// Returns the square of the euclidean distance between 2 points.
inline double dist(Point a, Point b)
{
	double x = (double)(a.x - b.x);
	double y = (double)(a.y - b.y);
	return (x * x) + (y * y);
}

// Returns the angle that the line from b to a makes with the line from b to c
inline uint32_t AngleABC(Point a, Point b, Point c)
{
	Point ab = {b.x - a.x, b.y - a.y};
	Point cb = {b.x - c.x, b.y - c.y};

	float dot = (float)(ab.x * cb.x + ab.y * cb.y);   // dot product
	float cross = (float)(ab.x * cb.y - ab.y * cb.x); // cross product

	float alpha = atan2(cross, dot);

	return (uint32_t)abs(floor(alpha * 180. / pi + 0.5));
}

void on_trackbar(int, void*)
{
}

GestureRecognition::GestureRecognition()
{
	try
	{
		cout << FACE_CASCADE_PATH << endl;
		cout << EYE_CASCADE_PATH << endl;
		m_face_cascade = cv::cuda::CascadeClassifier::create(FACE_CASCADE_PATH);
		m_eye_cascade = cv::cuda::CascadeClassifier::create(EYE_CASCADE_PATH);
	}
	catch (cv::Exception& e)
	{
		cout << e.what();
		cout << e.what();
	}
}

GestureRecognition::~GestureRecognition()
{
} // cvDestroyAllWindows(); }

DWORD WINAPI GestureRecognition::FindFists(LPVOID lpParam)
{
	ThreadInfo* info = (ThreadInfo*)lpParam;
	return 0;
}

DWORD WINAPI GestureRecognition::FindPalms(LPVOID lpParam)
{
	ThreadInfo* info = (ThreadInfo*)lpParam;
	return 0;
}

vector<GestureInfo> GestureRecognition::GetGestureInfo(cv::cuda::GpuMat image)
{

	cuda::GpuMat skin_mask(image.size(), CV_8UC1);
	SkinInfo skin_info;
	skin_info.face_cascade = &m_face_cascade;
	skin_info.image = image;
	skin_info.output_image = &skin_mask;
	HANDLE skin_handle = CreateThread(NULL, 0, FindPalms, &skin_info, 0, NULL);
	if (skin_handle == NULL)
	{
		// throw threading exception?
		return vector<GestureInfo>();
	}

	vector<GestureInfo> fists_detected;
	ThreadInfo fist_info;
	fist_info.gesture_info = &fists_detected;
	fist_info.image = image;
	fist_info.gest_ptr = this;
	HANDLE fist_handle = CreateThread(NULL, 0, FindFists, &fist_info, 0, NULL);
	if (fist_handle == NULL)
	{
		// throw threading exception?
		return vector<GestureInfo>();
	}

	vector<GestureInfo> palms_detected;
	ThreadInfo palm_info;
	palm_info.gesture_info = &palms_detected;
	palm_info.image = image;
	palm_info.gest_ptr = this;
	HANDLE palm_handle = CreateThread(NULL, 0, FindPalms, &palm_info, 0, NULL);
	if (palm_handle == NULL)
	{
		// throw threading exception?
		return vector<GestureInfo>();
	}

	// 1 second
	DWORD timeout = 1000;
	DWORD thread_code;
	if ((thread_code = WaitForSingleObject(fist_handle, timeout)) != WAIT_OBJECT_0)
	{
		if (thread_code == WAIT_TIMEOUT)
		{
			TerminateThread(fist_handle, 0);
		}
	}

	if ((thread_code = WaitForSingleObject(palm_handle, timeout)) != WAIT_OBJECT_0)
	{
		if (thread_code == WAIT_TIMEOUT)
		{
			TerminateThread(palm_handle, 0);
		}
	}

	if ((thread_code = WaitForSingleObject(skin_handle, timeout)) != WAIT_OBJECT_0)
	{
		if (thread_code == WAIT_TIMEOUT)
		{
			TerminateThread(skin_handle, 0);
		}
	}

	// Throw out any that have too small of an area
	for (vector<GestureInfo>::iterator fist_iter = fists_detected.begin(); fist_iter != fists_detected.end();)
	{
		if ((*fist_iter).rect.area() < MINIMUM_CONTOUR_AREA)
			fist_iter = fists_detected.erase(fist_iter);
		else
			++fist_iter;
	}
	for (vector<GestureInfo>::iterator palm_iter = palms_detected.begin(); palm_iter != palms_detected.end();)
	{
		if ((*palm_iter).rect.area() < MINIMUM_CONTOUR_AREA)
			palm_iter = palms_detected.erase(palm_iter);
		else
			++palm_iter;
	}

	for (vector<GestureInfo>::iterator fist_iter = fists_detected.begin(); fist_iter != fists_detected.end();
		 fist_iter++)
	{
		for (vector<GestureInfo>::iterator palm_iter = palms_detected.begin(); palm_iter != palms_detected.end();)
		{
			// For now if the palms/fists intersect, assume palm is correct
			if ((fist_iter->rect & palm_iter->rect).area() > 0)
				palm_iter = palms_detected.erase(palm_iter);
			else
				++palm_iter;
		}
	}

	vector<GestureInfo> gesture_info;
	for (size_t gest_idx = 0; gest_idx < palms_detected.size(); gest_idx++)
	{
		gesture_info.push_back(palms_detected[gest_idx]);
	}
	for (size_t gest_idx = 0; gest_idx < fists_detected.size(); gest_idx++)
	{
		gesture_info.push_back(fists_detected[gest_idx]);
	}

	vector<vector<Point>> contours = GetContours(&image);
	ContourExamination(contours, &gesture_info, &image);

	return gesture_info;
}

void GestureRecognition::ContourExamination(vector<vector<Point>>& contours,
											vector<GestureInfo>* gesture_info,
											cuda::GpuMat* image)
{
	// Remove contours that are too small
	for (vector<vector<Point>>::iterator contour_iterator = contours.begin(); contour_iterator != contours.end();)
	{
		if (fabs(contourArea(*contour_iterator)) < MINIMUM_CONTOUR_AREA)
			contour_iterator = contours.erase(contour_iterator);
		else
			++contour_iterator;
	}

	uint32_t hand_count = 0;
	for (size_t cont_idx = 0; cont_idx < contours.size(); cont_idx++)
	{
		// Check if any of the current gestures belong to the contour

		// If the contour is not registered as a fist, examine it
		bool gesture_detected = false;
		for (size_t gest_idx = 0; gest_idx < gesture_info->size(); gest_idx++)
		{
			if (pointPolygonTest(contours[cont_idx], (*gesture_info)[gest_idx].centroid, false) >= 0)
			{
				gesture_detected = true;
				break;
			}
		}

		if (gesture_detected)
		{
			continue;
		}

		/*vector<std::vector<int>> convex_hull(contours.size());
		convexHull(contours[cont_idx], convex_hull[cont_idx], false);

		vector<vector<Vec4i>> convexity_defects(contours.size());
		convexityDefects(contours[cont_idx], convex_hull[cont_idx], convexity_defects[cont_idx]);*/
		vector<Vec4i> defects = GetDefects(contours[cont_idx]);

		int average_finger_separation = 0;
		int average_finger_length = 0;
		int fingers_found = 0;
		Point lowest_defect;
		lowest_defect.y = INT_MAX;
		Point highest_defect;
		highest_defect.y = 0;
		Point leftmost_defect;
		leftmost_defect.x = INT_MAX;
		Point rightmost_defect;
		rightmost_defect.x = 0;

		// Remove contours that are too small
		for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();)
		{
			double defect_depth = (double)((*defect_iterator).val[3] / 256.0f);
			if (defect_depth < MINIMUM_DEFECT_DEPTH)
			{
				defect_iterator = defects.erase(defect_iterator);
				continue;
			}

			int start_index = defect_iterator->val[0];
			Point start_point(contours[cont_idx][start_index]);
			int end_index = defect_iterator->val[1];
			Point end_point(contours[cont_idx][end_index]);
			int far_index = defect_iterator->val[2];
			Point far_point(contours[cont_idx][far_index]);

			if (far_point.x > rightmost_defect.x)
			{
				rightmost_defect.x = far_point.x;
				rightmost_defect.y = far_point.y;
			}
			if (far_point.x < leftmost_defect.x)
			{
				leftmost_defect.x = far_point.x;
				leftmost_defect.y = far_point.y;
			}
			if (far_point.y > highest_defect.y)
			{
				highest_defect.x = far_point.x;
				highest_defect.y = far_point.y;
			}
			if (far_point.y < lowest_defect.y)
			{
				lowest_defect.x = far_point.x;
				lowest_defect.y = far_point.y;
			}

			uint32_t angle = AngleABC(start_point, far_point, end_point);
			if (angle < ANGLE_BETWEEN_FINGERS)
			{
				// If this is the first, double up because there must be another because of this detection technique
				fingers_found = (fingers_found ? fingers_found + 1 : fingers_found + 2);
				average_finger_separation += (int)dist(start_point, end_point);

				// could do some trigonometry to get the proper separation length possibly??
				average_finger_length += (int)(dist(far_point, start_point) + dist(far_point, end_point)) / 2;
#ifdef _DEBUG
				line(*image, start_point, far_point, COLOR_RED, 1, CV_AA, 0);
				circle(*image, far_point, 5, COLOR_TEAL, 2, 8, 0);
				circle(*image, start_point, 5, COLOR_PURPLE, 2, 8, 0);
				line(*image, start_point, end_point, COLOR_BLUE, 1, CV_AA, 0);
				line(*image, far_point, end_point, COLOR_GREEN, 1, CV_AA, 0);
#endif

				defect_iterator = defects.erase(defect_iterator);
				continue;
			}
			defect_iterator++;
		}

		GestureInfo gesture;
		if (fingers_found == 4)
		{
			gesture.thumb_extended = false;
			gesture.gesture = GestureDetected::FingersExtended;
		}
		else if (fingers_found == 2)
		{
			gesture.thumb_extended = false;
			if (average_finger_separation > average_finger_length)
			{
				gesture.gesture = GestureDetected::PinkyIndexExtended;
			}
			else
			{
				gesture.gesture = GestureDetected::MiddleIndexExtended;
			}
		}
		else
		{
			// Defect is not a recognized gesture
			continue;
		}

		bool thumb_found = false;

		average_finger_separation /= (fingers_found - 1);
		average_finger_length /= (fingers_found - 1);

		Point center_of_palm;
		gesture.rect.x = leftmost_defect.x;
		gesture.rect.width = (rightmost_defect.x - leftmost_defect.x);
		gesture.rect.y = highest_defect.y;
		gesture.rect.height = (highest_defect.y - lowest_defect.y);

		center_of_palm.x = (rightmost_defect.x + leftmost_defect.x) / 2;
		center_of_palm.y = (highest_defect.y + lowest_defect.y) / 2;
		gesture.centroid.x = center_of_palm.x;
		gesture.centroid.y = center_of_palm.y;

		int farthest, shortest, container;
		farthest = (int)dist(center_of_palm, highest_defect);
		shortest = (int)dist(center_of_palm, highest_defect);
		farthest = ((container = (int)dist(center_of_palm, rightmost_defect)) > farthest) ? container : farthest;
		shortest = (container < shortest) ? container : shortest;

		circle(*image, center_of_palm, (int)sqrt(shortest), COLOR_BLUE, 2, 8, 0);

		for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();
			 defect_iterator++)
		{
			int start_index = defect_iterator->val[0];
			Point start_point(contours[cont_idx][start_index]);
			int end_index = defect_iterator->val[1];
			Point end_point(contours[cont_idx][end_index]);
			int far_index = defect_iterator->val[2];
			Point far_point(contours[cont_idx][far_index]);

			uint32_t angle = AngleABC(start_point, far_point, end_point);
			if (angle >= ANGLE_BETWEEN_FINGERS)
			{
				// Could be a thumb, check the length.
				int length = (int)(dist(far_point, end_point) + dist(far_point, start_point)) / 2;
				if (length > (2 * average_finger_length))
				{
#ifdef _DEBUG
					line(*image, start_point, far_point, COLOR_DARK_RED, 1, CV_AA, 0);
					circle(*image, far_point, 5, COLOR_DARK_TEAL, 2, 8, 0);
					circle(*image, start_point, 5, COLOR_DARK_PURPLE, 2, 8, 0);
					line(*image, start_point, end_point, COLOR_DARK_BLUE, 1, CV_AA, 0);
					line(*image, far_point, end_point, COLOR_DARK_GREEN, 1, CV_AA, 0);
#endif
				}
				else
				{
					gesture.thumb_extended = true;
					thumb_found = true;
#ifdef _DEBUG
					line(*image, start_point, far_point, COLOR_YELLOW, 1, CV_AA, 0);
					circle(*image, far_point, 5, COLOR_DARK_YELLOW, 2, 8, 0);
					circle(*image, start_point, 5, COLOR_BROWN, 2, 8, 0);
					line(*image, start_point, end_point, COLOR_PINK, 1, CV_AA, 0);
					line(*image, far_point, end_point, COLOR_AQUA, 1, CV_AA, 0);
#endif
				}
			}
		}

		gesture_info->push_back(gesture);
	}
}

// This should probably return information about the palm, maybe extreme
// points of contour extraction
/*bool GestureRecognition::SkinAnalysis(cuda::GpuMat* image, vector<Rect> fists, vector<Rect> palms)
{
	int actual_contours = 0;

	// Extract contours of image
	vector<vector<Point>> contours;
	if (FindContours(image, &contours) <= 0)
	{
		return false; // Could not find anything
	}

	// Remove contours that are too small
	for (vector<vector<Point>>::iterator contour_iterator = contours.begin(); contour_iterator != contours.end();)
	{
		if (fabs(contourArea(*contour_iterator)) < MINIMUM_CONTOUR_AREA)
			contour_iterator = contours.erase(contour_iterator);
		else
			++contour_iterator;
	}

	uint32_t hand_count = 0;
	for (size_t contour_idx = 0; contour_idx < contours.size(); contour_idx++)
	{
		// Check if any of the current gestures belong to the contour

		// If the contour is not registered as a fist, examine it
		vector<Vec4i> defects = GetDefects(contours[contour_idx]);

		int average_finger_separation = 0;
		int average_finger_length = 0;
		int fingers_found = 0;
		Point lowest_defect;
		lowest_defect.y = (*image).size().height;
		Point highest_defect;
		highest_defect.y = 0;
		Point leftmost_defect;
		leftmost_defect.x = (*image).size().width;
		Point rightmost_defect;
		rightmost_defect.x = 0;

		// Remove contours that are too small
		for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();)
		{
			double defect_depth = (double)((*defect_iterator).val[3] / 256.0f);
			if (defect_depth < MINIMUM_DEFECT_DEPTH)
			{
				defect_iterator = defects.erase(defect_iterator);
				continue;
			}

			int start_index = (*defect_iterator).val[0];
			Point start_point(contours[contour_idx][start_index]);
			int end_index = (*defect_iterator).val[1];
			Point end_point(contours[contour_idx][end_index]);
			int far_index = (*defect_iterator).val[2];
			Point far_point(contours[contour_idx][far_index]);

			if (far_point.x > rightmost_defect.x)
			{
				rightmost_defect.x = far_point.x;
				rightmost_defect.y = far_point.y;
			}
			if (far_point.x < leftmost_defect.x)
			{
				leftmost_defect.x = far_point.x;
				leftmost_defect.y = far_point.y;
			}
			if (far_point.y > highest_defect.y)
			{
				highest_defect.x = far_point.x;
				highest_defect.y = far_point.y;
			}
			if (far_point.y < lowest_defect.y)
			{
				lowest_defect.x = far_point.x;
				lowest_defect.y = far_point.y;
			}

			uint32_t angle = AngleABC(start_point, far_point, end_point);
			if (angle < ANGLE_BETWEEN_FINGERS)
			{
				fingers_found++;
				average_finger_separation += (int)dist(start_point, end_point);
				average_finger_length += (int)(dist(far_point, start_point) + dist(far_point, end_point)) / 2;
#ifdef _DEBUG
				line(*image, start_point, far_point, COLOR_RED, 1, CV_AA, 0);
				circle(*image, far_point, 5, COLOR_TEAL, 2, 8, 0);
				circle(*image, start_point, 5, COLOR_PURPLE, 2, 8, 0);
				line(*image, start_point, end_point, COLOR_BLUE, 1, CV_AA, 0);
				line(*image, far_point, end_point, COLOR_GREEN, 1, CV_AA, 0);
#endif

				defect_iterator = defects.erase(defect_iterator);
				continue;
			}
			defect_iterator++;
		}

		if (fingers_found)
		{
			hand_count++;
		}
		else
		{
			continue;
		}

		bool thumb_found = false;

		average_finger_separation /= fingers_found;
		average_finger_length /= fingers_found;

		Point center_of_palm;
		center_of_palm.x = (rightmost_defect.x + leftmost_defect.x) / 2;
		center_of_palm.y = (highest_defect.y + lowest_defect.y) / 2;

		vector<int> palm_distances;
		palm_distances.push_back((int)dist(center_of_palm, highest_defect));
		palm_distances.push_back((int)dist(center_of_palm, lowest_defect));
		palm_distances.push_back((int)dist(center_of_palm, rightmost_defect));
		palm_distances.push_back((int)dist(center_of_palm, leftmost_defect));
		int farthest = palm_distances[0];
		int shortest = palm_distances[0];

		// can start loop at 1 because farthest and shortest are loaded with idx 0
		for (size_t palm_dist_idx = 1; palm_dist_idx < palm_distances.size(); palm_dist_idx++)
		{
			if (palm_distances[palm_dist_idx] > farthest)
			{
				farthest = palm_distances[palm_dist_idx];
			}
			if (palm_distances[palm_dist_idx] < shortest)
			{
				shortest = palm_distances[palm_dist_idx];
			}
		}
		circle(*image, center_of_palm, (int)sqrt(shortest), COLOR_BLUE, 2, 8, 0);

		namedWindow("palm", 1);
		imshow("palm", *image);
		waitKey(-1);

		for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();
			 defect_iterator++)
		{
			int start_index = (*defect_iterator).val[0];
			Point start_point(contours[contour_idx][start_index]);
			int end_index = (*defect_iterator).val[1];
			Point end_point(contours[contour_idx][end_index]);
			int far_index = (*defect_iterator).val[2];
			Point far_point(contours[contour_idx][far_index]);

			uint32_t angle = AngleABC(start_point, far_point, end_point);
			if (angle > ANGLE_BETWEEN_FINGERS)
			{
				// Could be a thumb, check the length.
				int length = (int)(dist(far_point, end_point) + dist(far_point, start_point)) / 2;
				if (length > (2 * average_finger_length))
				{
#ifdef _DEBUG
					line(*image, start_point, far_point, COLOR_DARK_RED, 1, CV_AA, 0);
					circle(*image, far_point, 5, COLOR_DARK_TEAL, 2, 8, 0);
					circle(*image, start_point, 5, COLOR_DARK_PURPLE, 2, 8, 0);
					line(*image, start_point, end_point, COLOR_DARK_BLUE, 1, CV_AA, 0);
					line(*image, far_point, end_point, COLOR_DARK_GREEN, 1, CV_AA, 0);
#endif
				}
				else
				{
					thumb_found = true;
#ifdef _DEBUG
					line(*image, start_point, far_point, COLOR_YELLOW, 1, CV_AA, 0);
					circle(*image, far_point, 5, COLOR_DARK_YELLOW, 2, 8, 0);
					circle(*image, start_point, 5, COLOR_BROWN, 2, 8, 0);
					line(*image, start_point, end_point, COLOR_PINK, 1, CV_AA, 0);
					line(*image, far_point, end_point, COLOR_AQUA, 1, CV_AA, 0);
#endif
				}
			}
		}
	}

#ifdef _DEBUG
	namedWindow("Final", 1);
	imshow("Final", *image);
	waitKey(-1);
#endif
	return ((actual_contours > 0) || (actual_contours < 5));
}*/

vector<vector<Point>> GestureRecognition::GetContours(cv::cuda::GpuMat* image)
{
	// Find contours
	vector<Vec4i> hierarchy;
	vector<vector<Point>> contours;
	findContours(*image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	return contours;
}

std::vector<cv::Vec4i> GestureRecognition::GetDefects(std::vector<cv::Point>& contours)
{
	vector<int> hull;
	vector<Vec4i> defects;
	convexHull(contours, hull, false);
	convexityDefects(contours, hull, defects);
	return defects;
}

DWORD WINAPI GestureRecognition::SkinExtraction(LPVOID lpParam)
{
	SkinInfo* skin_info = (SkinInfo*)lpParam;
	cuda::GpuMat* image = &(skin_info->image);
	Mat skin_extraction;
	(*image).download(skin_extraction);
	Mat skin_extraction_shrunk;
	blur(skin_extraction, skin_extraction, Size(4, 4));
	// resize(skin_extraction, skin_extraction_shrunk, Size(640, 480), 0, 0, CV_INTER_AREA);

	Mat gray_mat;
	cvtColor(skin_extraction, gray_mat, CV_BGR2GRAY);
	equalizeHist(gray_mat, gray_mat);

	// Convert to GpuMat for cuda acceleration
	cv::cuda::GpuMat gray_gpumat;
	gray_gpumat.upload(gray_mat);

	// Detect all the faces in the image
	cv::cuda::GpuMat face_mat;
	std::vector<Rect> faces;
	(*(skin_info->face_cascade))->detectMultiScale(gray_gpumat, gray_gpumat);
	(*(skin_info->face_cascade))->convert(gray_gpumat, faces);

	if (!faces.size())
	{
		// Should perform static skin detection in this case..... ?? ?? ? ? ? ? ? ? ? ? ? ??????
		return 1;
	}

	// Generate YCrCb representation of image
	Mat ycrcb_mat;
	cvtColor(skin_extraction, ycrcb_mat, CV_BGR2YCrCb);
	normalize(ycrcb_mat, ycrcb_mat, 0, 255, NORM_MINMAX);

	// Generate vector of faces in image
	std::vector<Mat> face_ycrcb;
	for (size_t face_idx = 0; face_idx < faces.size(); face_idx++)
	{
		Rect face_shrunk;
		face_shrunk.x = faces[face_idx].x + (int)(0.30 * faces[face_idx].width);
		face_shrunk.width = (int)(0.40 * faces[face_idx].width); //
		face_shrunk.y = faces[face_idx].y;						 // -0.10 * faces[face_idx].y;
		face_shrunk.height = faces[face_idx].height;
		face_ycrcb.push_back(ycrcb_mat(face_shrunk));
	}

	// General histogram parameters... Use all 3 channels, with max size bins
	int ycrcb_channels[] = {1, 2};
	int channel_count = 2;
	int image_count = 1;
	int bins_cnt = 2;
	int hist_size[] = {122, 122};
	bool accumulate = false;
	bool uniform = true;

	// YCrCb information, allow max ranges
	float yranges[] = {0, 255};
	float crranges[] = {0, 255}; // {130, 180}; // 138 to 173
	float cbranges[] = {0, 255}; //  {60, 140}; // 67 to 133
	const float* ycrcb_ranges[] = {crranges, cbranges};

	// Create back projection vectors
	std::vector<Mat> hsv_backproj_vec, ycrcb_backproj_vec;

	for (size_t ycrcb_idx = 0; ycrcb_idx < face_ycrcb.size(); ycrcb_idx++)
	{
		// Generate histogram
		Mat ycrcb_hist;
		calcHist(&(face_ycrcb[ycrcb_idx]),
				 image_count,
				 ycrcb_channels,
				 Mat(),
				 ycrcb_hist,
				 2,
				 hist_size,
				 ycrcb_ranges,
				 uniform,
				 accumulate);

		// Get back projection
		Mat ycrcb_backproj;
		calcBackProject(&ycrcb_mat, image_count, ycrcb_channels, ycrcb_hist, ycrcb_backproj, ycrcb_ranges, 1, true);
		ycrcb_backproj_vec.push_back(ycrcb_backproj);
	}

	// Get highest change out of all mats that pixel is skin

	Mat highest_probability(ycrcb_backproj_vec[0]);

	for (size_t ycrcb_idx = 1; ycrcb_idx < ycrcb_backproj_vec.size(); ycrcb_idx++)
	{
		MatIterator_<unsigned char> pixel_iter, ycrcb_iter, end;
		for (pixel_iter = highest_probability.begin<unsigned char>(),
			ycrcb_iter = ycrcb_backproj_vec[ycrcb_idx].begin<unsigned char>(),
			end = highest_probability.end<unsigned char>();
			 pixel_iter != end;
			 ++pixel_iter, ++ycrcb_iter)
		{
			if (*pixel_iter < *ycrcb_iter)
			{
				*pixel_iter = *ycrcb_iter;
			}
		}
	}

	Mat output_mat(highest_probability.size(), CV_8UC1);
	threshold(highest_probability, output_mat, (double)MIN_CHANCE_80, 255, CV_THRESH_BINARY);

	Mat final_output(highest_probability.size(), CV_8UC1);
	Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(11, 11));
	erode(output_mat, final_output, kernel);
	dilate(final_output, final_output, kernel);

	skin_info->output_image->upload(final_output);

	return 0;
}

// Entry point
int main()
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())
	{
		IMAGE_GRAB_EXCEPTION("Failed to open camera.");
	}
	uint64_t count = 0;
	Mat image;

	for (uint32_t i = 0; i < 15; i++)
	{
		if (!cap.read(image))
		{
			IMAGE_GRAB_EXCEPTION("Failed to grab first image.");
		}
	}

	cout << image.size() << endl;
	cuda::GpuMat gpu_image;

	GestureRecognition gesture_recognition;
	while (1)
	{
		try
		{
			// read img to pass to class
			vector<string> images_to_process;
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\green_wall_both_hands_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blidns_and_light_one_hand_fist.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_1_hand_middle_finger.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_and_light_both_hands_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_and_light_one_hand_peace.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_andLight_one_hand_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_both_hands_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_one_hand_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_one_hand_fist.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_one_hand_peace.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_one_hand_rockon.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\blinds_one_hand_rockon_thumb.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\brown_wall_both_hands_fist.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\brown_wall_both_hands_peace.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\brown_wall_one_hand_extended_sideways.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\brown_wall_one_hand_fist.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\green_wall_both_hands_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\variable_background_both_hands_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\green_wall_both_hands_peace.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\variable_background_one_hand_extended.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\green_wall_fists.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\variable_background_one_hand_fist.jpg");
			images_to_process.push_back("C:\\Users\\lyndo\\Downloads\\Training_set_"
										"pictures\\variable_background_one_hand_extended_other_sideways.jpg");
			/*if (!cap.read(image))
			{
				IMAGE_GRAB_EXCEPTION("Failed to grab image.");
			}*/

			// cout << "Image (" << image.size << ") " << image.type() << endl;
			// imshow("with face", image);
			for (size_t img_idx = 0; img_idx < images_to_process.size(); img_idx++)
			{
				Mat image_jpg;
				cout << "Image: " << images_to_process[img_idx] << endl;
				image_jpg = imread(images_to_process[img_idx].c_str(), CV_LOAD_IMAGE_COLOR);
				gpu_image.upload(image_jpg);
				gesture_recognition.GetGestureInfo(gpu_image);
			}

			// read extra image to flush buffer
			/*if (!cap.read(image))
			{
				IMAGE_GRAB_EXCEPTION("Failed to grab image.");
			}*/
		}
		catch (cv::Exception& e)
		{
			cout << "OpenCV exception: " << e.what();
			break;
		}
		catch (MemoryException& e)
		{
			cout << "Memory exception: " << e.what();
			break;
		}
		catch (ImageGrabException& e)
		{
			cout << "Image grab exception: " << e.what();
			break;
		}
		catch (FaceSubtractionException& e)
		{
			cout << "Face subtraction exception: " << e.what();
			break;
		}
	}

	cout << "Exception caused ending of program." << endl;
	return 0;
}
