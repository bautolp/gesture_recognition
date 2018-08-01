/* This Gesture recognition software is provided by Lyndon Bauto (github.com/bautolp) "as is" and "with all faults." I
 * make no representations or warranties of any kind concerning the safety, suitability, lack of viruses,
 * inaccuracies, typographical errors, or other harmful components of this Gesture recognition software. There are
 * inherent dangers in the use of any software, and you are solely responsible for determining whether this  Gesture
 * recognition software is compatible with your equipment and other software installed on your equipment.You are also
 * solely responsible for the protection of your equipment and backup of your data, and I will not be liable
 * for any damages you may suffer in connection with using, modifying, or distributing this Gesture recognition
 * software. */

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
#include <thread>

// Maybe make runtime options and load the class with them
// Could allow cmd line adjustment of runtime options dynamically
// for performance tweaking
#define MINIMUM_CONTOUR_AREA 5000.0
#define MINIMUM_DEFECT_DEPTH 25.0
#define ANGLE_BETWEEN_FINGERS 55
#define ANGLE_BETWEEN_THUMB 120

#define X_WITHIN_Y_AMT(x, y, amt) (((x) >= ((y) - (amt)) && ((x) <= ((y) + (amt)))))

#define TUNING_SKIN_DETECTION
// General histogram parameters... Use all 3 channels, with max size bins
int ycrcb_channels[] = {1, 2};
int channel_count = 2;
int image_count = 1;
#define CR_MIN 138
#define CR_MAX 178
#define CB_MIN 62
#define CB_MAX 138
#define CR_SIZE (CR_MAX - CR_MIN)
#define CB_SIZE (CB_MAX - CB_MIN)
int hist_size[] = {CR_SIZE / 2, CB_SIZE / 2};
// YCrCb information, allow max ranges
float yranges[] = {0, 255};
float crranges[] = {CR_MIN, CR_MAX}; // {130, 180}; // 138 to 173 // at school before: 143 - 173
float cbranges[] = {CB_MIN, CB_MAX}; //  {60, 140}; // 67 to 133 // at schoole before: 80 - 125
const float* ycrcb_ranges[] = {crranges, cbranges};

#ifndef NO_LOG
#define DISPLAY_AND_ANNOTATE
#define LOG_COUT
#define WRITE_VIDEO
#endif
#define MIN_CHANCE(amt) ((unsigned char)(((float)(amt) / (float)100) * (float)255))

#ifndef pi
#define pi 3.14159265359
#endif

#define FACE_CASCADE_PATH                                                                                              \
	"C:\\Users\\lyndo\\Developer\\OpenCV_"                                                                             \
	"cuda\\install\\etc\\haarcascades\\haarcascade_"                                                                   \
	"frontalface_alt.xml"

#define SKIN_DETECTION_FREQUENCY 1

using namespace cv;
using namespace cuda;
using namespace std;
using namespace chrono;

// Returns the square of the euclidean distance between 2 points.
inline double dist(Point a, Point b)
{
	double x = (double)(a.x - b.x);
	double y = (double)(a.y - b.y);
	return (x * x) + (y * y);
}

// I think this has a bug, saw angles > 4k coming out???? WTF
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

GestureRecognition::GestureRecognition()
{
	try
	{
		m_cuda_stream = new mutex;
		cout << FACE_CASCADE_PATH << endl;
		m_face_cascade = cv::cuda::CascadeClassifier::create(FACE_CASCADE_PATH);
	}
	catch (cv::Exception& e)
	{
		cout << e.what();
		cout << e.what();
	}
}

GestureRecognition::~GestureRecognition()
{
	delete m_cuda_stream;
} // cvDestroyAllWindows(); }

vector<Gesture> GestureRecognition::GetGestureInfo(cv::cuda::GpuMat image, steady_clock::time_point timestamp)
{
	Mat display_img(image.size(), image.type());
	image.download(display_img);
	Mat output_display;
	resize(display_img, display_img, Size(), 0.5, 0.5, INTER_NEAREST);
	bilateralFilter(display_img, output_display, 8, 500, 500);
	image.upload(output_display);
	SkinInfo skin_info;
	skin_info.face_cascade = &m_face_cascade;
	skin_info.image = image;
	skin_info.cuda_stream = m_cuda_stream;
	SkinExtraction(&skin_info);

	// Should profile this time and if it's high, thread
	vector<Gesture> gestures;
	for (size_t index = 0; index < skin_info.face_info.size(); index++)
	{
#ifdef DISPLAY_AND_ANNOTATE
		/*namedWindow("Mask" + to_string(index), CV_WINDOW_NORMAL);
		imshow("Mask" + to_string(index), skin_info.face_info[index].skin_mask);
		waitKey(10);
		rectangle(output_display, skin_info.face_info[index].face_rect, Scalar(255, 255, 0), 3, 8, 0);*/
#endif
		vector<vector<Point>> contours = GetContours(&skin_info.face_info[index].skin_mask);
		ContourExamination(contours, &gestures, skin_info.face_info[index].face_rect, output_display);
	}
	for (size_t gest_idx = 0; gest_idx < gestures.size(); gest_idx++)
	{
		gestures[gest_idx].SetTime(timestamp);
	}
#ifdef DISPLAY_AND_ANNOTATE
	namedWindow("Input", CV_WINDOW_NORMAL);
	imshow("Input", output_display);
	waitKey(10);
#else
	Mat display_img;
#endif
	return gestures;
}

void GestureRecognition::DetermineGesture(Gesture* gesture, int fing_cnt)
{
	switch (fing_cnt)
	{
		case 4:
			gesture->SetFingerCount(FingerCount::FourFingers);
			break;
		case 3:
			gesture->SetFingerCount(FingerCount::ThreeFingers);
			break;
		case 2:
			gesture->SetFingerCount(FingerCount::TwoFingers);
			break;
		case 1:
			gesture->SetFingerCount(FingerCount::OneFinger);
			break;
		case 0:
			gesture->SetFingerCount(FingerCount::Fist);
			break;
		default:
			gesture->SetFingerCount(FingerCount::Invalid);
			break;
	}
}

void GestureRecognition::AddIfNotInVector(vector<Point>* points, Point& point, int face_width, int* finger_count)
{
	for (size_t point_idx = 0; point_idx < points->size(); point_idx++)
	{
		if (point.x >= ((*points)[point_idx].x - face_width * 0.175) &&
			point.x <= ((*points)[point_idx].x + face_width * 0.175) &&
			point.y >= ((*points)[point_idx].y - face_width * 0.175) &&
			point.y <= ((*points)[point_idx].y + face_width * 0.175))
		{
			return;
		}
	}
	points->push_back(point);
	*finger_count = *finger_count + 1;
}

void GestureRecognition::FingerDetection(vector<Point>& contours,
										 Gesture& gesture,
										 vector<Vec4i>& defects,
										 Mat& image,
										 Rect& face,
										 Point& hi_point,
										 Point& l_point,
										 Point& r_point)
{
	int finger_count = 0;
	bool between_found = false;
	bool single_finger_found = false;
	bool single_finger_found_twice = false;
	vector<Point> counted_points;
	gesture.SetFingerCount(FingerCount::Fist);

	Point l_near_hi = hi_point;
	Point r_near_hi = hi_point;
	Point r_above_wrist = hi_point;
	Point l_above_wrist = hi_point;

	FilterDefects(defects, face, contours, hi_point);
	// Find contour points that are near hi point vertically, but further left or further right
	int width_for_check = (int)(face.height * 0.15);
	for (size_t idx = 0; idx < contours.size(); idx++)
	{
		if (X_WITHIN_Y_AMT(contours[idx].y, hi_point.y, width_for_check))
		{
			if (r_near_hi.x < contours[idx].x)
				r_near_hi = contours[idx];
			if (l_near_hi.x > contours[idx].x)
				l_near_hi = contours[idx];
		}
		if (contours[idx].y < (hi_point.y + face.height * 1.5))
		{
			if (r_above_wrist.x < contours[idx].x)
				r_above_wrist = contours[idx];
			if (l_above_wrist.x > contours[idx].x)
				l_above_wrist = contours[idx];
		}
	}

	bool is_thumb = false;
	bool is_reverse_thumb = false;
	HandSide side = (face.x > hi_point.x) ? Left : Right;
	int height_for_check = face.height * 0.175;
	Point hi_near_thumb, lo_near_thumb;
	if (side == Left)
	{
		hi_near_thumb = r_above_wrist;
		lo_near_thumb = r_above_wrist;
		for (size_t idx = 0; idx < contours.size(); idx++)
		{
			// Loop through all the points and check the highest and lowest)
			if (contours[idx].x > hi_point.x)
			{
				if ((contours[idx].y > (r_above_wrist.y - height_for_check)) && (contours[idx].y < r_above_wrist.y))
				{
					if (contours[idx].x < hi_near_thumb.x)
					{
						hi_near_thumb = contours[idx];
					}
				}
				else if ((contours[idx].y < (r_above_wrist.y + height_for_check)) &&
						 (contours[idx].y > r_above_wrist.y))
				{
					if (contours[idx].x < lo_near_thumb.x)
					{
						lo_near_thumb = contours[idx];
					}
				}
			}
		}
		is_thumb = ((hi_near_thumb.x < (r_above_wrist.x - (face.width * 0.4))) &&
					(lo_near_thumb.x < (r_above_wrist.x - (face.width * 0.4))));
		hi_near_thumb = l_above_wrist;
		lo_near_thumb = l_above_wrist;
		for (size_t idx = 0; idx < contours.size(); idx++)
		{
			if (contours[idx].x < hi_point.x)
			{
				// Loop through all the points and check the highest and lowest)
				if ((contours[idx].y > (l_above_wrist.y - height_for_check)) && (contours[idx].y < l_above_wrist.y))
				{
					if (contours[idx].x > hi_near_thumb.x)
					{
						hi_near_thumb = contours[idx];
					}
				}
				else if ((contours[idx].y < (l_above_wrist.y + height_for_check)) &&
						 (contours[idx].y > l_above_wrist.y))
				{
					if (contours[idx].x > lo_near_thumb.x)
					{
						lo_near_thumb = contours[idx];
					}
				}
			}
		}
		is_reverse_thumb = ((hi_near_thumb.x > (l_above_wrist.x + (face.width * 0.4))) &&
							(lo_near_thumb.x > (l_above_wrist.x + (face.width * 0.4))));
	}
	else
	{
		hi_near_thumb = l_above_wrist;
		lo_near_thumb = l_above_wrist;
		for (size_t idx = 0; idx < contours.size(); idx++)
		{
			if (contours[idx].x < hi_point.x)
			{
				// Loop through all the points and check the highest and lowest)
				if ((contours[idx].y > (l_above_wrist.y - height_for_check)) && (contours[idx].y < l_above_wrist.y))
				{
					if (contours[idx].x > hi_near_thumb.x)
					{
						hi_near_thumb = contours[idx];
					}
				}
				else if ((contours[idx].y < (l_above_wrist.y + height_for_check)) &&
						 (contours[idx].y > l_above_wrist.y))
				{
					if (contours[idx].x > lo_near_thumb.x)
					{
						lo_near_thumb = contours[idx];
					}
				}
			}
		}
		is_thumb = ((hi_near_thumb.x > (l_above_wrist.x + (face.width * 0.4))) &&
					(lo_near_thumb.x > (l_above_wrist.x + (face.width * 0.4))));
		hi_near_thumb = r_above_wrist;
		lo_near_thumb = r_above_wrist;
		for (size_t idx = 0; idx < contours.size(); idx++)
		{
			// Loop through all the points and check the highest and lowest)
			if (contours[idx].x > hi_point.x)
			{
				if ((contours[idx].y > (r_above_wrist.y - height_for_check)) && (contours[idx].y < r_above_wrist.y))
				{
					if (contours[idx].x < hi_near_thumb.x)
					{
						hi_near_thumb = contours[idx];
					}
				}
				else if ((contours[idx].y < (r_above_wrist.y + height_for_check)) &&
						 (contours[idx].y > r_above_wrist.y))
				{
					if (contours[idx].x < lo_near_thumb.x)
					{
						lo_near_thumb = contours[idx];
					}
				}
			}
		}
		is_reverse_thumb = ((hi_near_thumb.x < (r_above_wrist.x - (face.width * 0.4))) &&
							(lo_near_thumb.x < (r_above_wrist.x - (face.width * 0.4))));
	}

	gesture.SetSide(side);
	gesture.SetThumb(In);
	gesture.SetFingerCount(Fist);
	for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();)
	{
		Point start_point(contours[defect_iterator->val[0]]);
		Point end_point(contours[defect_iterator->val[1]]);
		Point far_point(contours[defect_iterator->val[2]]);

		// Check if far point is a decent bit higher than lower of start point and end point, if so toss it

		// Get leftmost and rightmost points... Need to check their distance to far point
		Point left(INT_MAX, 0);
		Point right(0, 0);
		right = (start_point.x > right.x) ? start_point : right;
		right = (end_point.x > right.x) ? end_point : right;
		left = (start_point.x < left.x) ? start_point : left;
		left = (end_point.x < left.x) ? end_point : left;
		// If vertical distance between start point and end point is about a finger tip length, we assume it is
		// between two fingers
		if (IsTwoFingers(start_point, end_point, face))
		{
			// Have we counted these finger tips yet?
			AddIfNotInVector(&counted_points, start_point, face.width, &finger_count);
			AddIfNotInVector(&counted_points, end_point, face.width, &finger_count);
			DisplayPoints(image, start_point, far_point, end_point, COLOR_GREEN, COLOR_GREEN, COLOR_GREEN);
		}
		else if (IsThumb(
					 side, right, left, r_near_hi, l_near_hi, r_above_wrist, l_above_wrist, hi_point, face, is_thumb))
		{
			if (gesture.GetThumb() == Out)
			{
				if (side == Left)
					AddIfNotInVector(&counted_points, right, face.width, &finger_count);
				else
					AddIfNotInVector(&counted_points, left, face.width, &finger_count);
			}
			gesture.SetThumb(Out);

			DisplayPoints(image, start_point, far_point, end_point, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE);
		}
		// If vertical distance between start and end point is large and it is not thumb, it is likely single finger
		else if (IsPinkyFinger(side, right, left, far_point, r_above_wrist, l_above_wrist, hi_point, face))
		{
			(side == Left) ? DisplayPoints(image, left, far_point, left, COLOR_RED, COLOR_RED, COLOR_RED) :
							 DisplayPoints(image, right, far_point, right, COLOR_RED, COLOR_RED, COLOR_RED);

			(side == Left) ? AddIfNotInVector(&counted_points, left, face.width, &finger_count) :
							 AddIfNotInVector(&counted_points, right, face.width, &finger_count);

			// Sometimes medium-large vertical gap between the points... This is caused when pinky and index make a
			// larger gap than expected (hand can be slightly tiled)
			if (IsTwoWithLargeGap(side, left, right))
			{
				// DisplayPoints(image, start_point, far_point, end_point, COLOR_AQUA, COLOR_DARK_YELLOW,
				// COLOR_PINK);
				(side == Left) ? DisplayPoints(image, right, right, right, COLOR_YELLOW, COLOR_YELLOW, COLOR_YELLOW) :
								 DisplayPoints(image, left, left, left, COLOR_YELLOW, COLOR_YELLOW, COLOR_YELLOW);
				(side == Left) ? AddIfNotInVector(&counted_points, right, face.width, &finger_count) :
								 AddIfNotInVector(&counted_points, left, face.width, &finger_count);
			}
		}
		else if (IsReverseThumb(side,
								right,
								left,
								r_near_hi,
								l_near_hi,
								r_above_wrist,
								l_above_wrist,
								hi_point,
								face,
								is_reverse_thumb))
		{
			if (gesture.GetThumb() == Out)
			{
				if (side == Left)
					AddIfNotInVector(&counted_points, left, face.width, &finger_count);
				else
					AddIfNotInVector(&counted_points, right, face.width, &finger_count);
			}
			gesture.SetThumb(Out);

			DisplayPoints(image, start_point, far_point, end_point, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE);
		}
		// Generic single finger handler
		else
		{
			Point higher = (start_point.y < end_point.y) ? start_point : end_point;
			AddIfNotInVector(&counted_points, higher, face.width, &finger_count);
			DisplayPoints(image, higher, higher, higher, COLOR_PINK, COLOR_PINK, COLOR_PINK);
			DisplayPoints(image, start_point, far_point, end_point, COLOR_PINK, COLOR_PINK, COLOR_PINK);
		}

		defect_iterator = defects.erase(defect_iterator);
		continue;
	}

	if (!gesture.GetGesture()->fingers == FingerCount::Invalid)
		DetermineGesture(&gesture, finger_count);
}

bool GestureRecognition::IsThumb(HandSide side,
								 Point& right_pt,
								 Point& left_pt,
								 Point& right_near_hi_pt,
								 Point& left_near_hi_pt,
								 Point& right_above_wrist,
								 Point& left_above_wrist,
								 Point& hi_point,
								 Rect& face,
								 bool thumb_exists)
{
	if (!thumb_exists)
	{
		return false;
	}
	switch (side)
	{
		case Right:
			if ( //(left_pt.x <= (left_near_hi_pt.x - face.width * 0.25)) &&
				(left_pt.x <= (left_above_wrist.x + face.width * 0.075)) &&
				(left_pt.x >= (left_above_wrist.x - (face.width * 0.075))))
			{
				return true;
			}
			break;
		case Left:
			if ( //(right_pt.x >= (right_near_hi_pt.x + face.width * 0.25)) &&
				(right_pt.x >= (right_above_wrist.x - face.width * 0.105)) &&
				(right_pt.x <= (right_above_wrist.x + face.width * 0.105)))
			{
				return true;
			}
			break;
	}
	return false;
}

bool GestureRecognition::IsReverseThumb(HandSide side,
										Point& right_pt,
										Point& left_pt,
										Point& right_near_hi_pt,
										Point& left_near_hi_pt,
										Point& right_above_wrist,
										Point& left_above_wrist,
										Point& hi_point,
										Rect& face,
										bool thumb_exists)
{
	if (!thumb_exists)
	{
		return false;
	}
	switch (side)
	{
		case Left:
			if ( //(left_pt.x <= (left_near_hi_pt.x - face.width * 0.25)) &&
				(left_pt.x <= (left_above_wrist.x + face.width * 0.075)) &&
				(left_pt.x >= (left_above_wrist.x - (face.width * 0.075))))
			{
				return true;
			}
			break;
		case Right:
			if ( //(right_pt.x >= (right_near_hi_pt.x + face.width * 0.25)) &&
				(right_pt.x >= (right_above_wrist.x - face.width * 0.075)) &&
				(right_pt.x <= (right_above_wrist.x + face.width * 0.075)))
			{
				return true;
			}
			break;
	}
	return false;
}

bool GestureRecognition::IsPinkyFinger(HandSide side,
									   Point& right_pt,
									   Point& left_pt,
									   Point& far_pt,
									   Point& right_above_wrist,
									   Point& left_above_wrist,
									   Point& hi_point,
									   Rect& face)
{
	int x, y;
	unsigned int angle;
	switch (side)
	{
		case Right:
			x = (right_above_wrist.x - face.width * 0.01);
			y = (hi_point.y + face.height * 0.75);
			angle = 70;
			return ((right_pt.x >= x) && (right_pt.y <= y) && (AngleABC(left_pt, far_pt, right_pt) < angle));
		case Left:
			x = (left_above_wrist.x + face.width * 0.1);
			y = (hi_point.y + face.height * 0.75);
			angle = 70;
			// If head is on the right, check if we're looking at the leftmost point, if so, we are likely looking
			// at the pinky in extended gesture (aLso check for rational height)
			return ((left_pt.x <= x) && (left_pt.y <= y) && (AngleABC(left_pt, far_pt, right_pt) < angle));
	}
	return false;
}
bool GestureRecognition::IsTwoWithLargeGap(HandSide side, cv::Point& left, cv::Point& right)
{
	switch (side)
	{
		case Left:
			return (left.y < right.y);
		case Right:
			return (right.y < left.y);
	}
	return false;
}
bool GestureRecognition::IsTwoFingers(cv::Point& start_pt, cv::Point& end_pt, cv::Rect& face)
{
	return X_WITHIN_Y_AMT(start_pt.y, end_pt.y, (face.height * 0.275));
}

bool GestureRecognition::ContourIsFace(std::vector<cv::Point>& contours, cv::Rect& face, cv::Mat& image)
{
	size_t point_cnt = 10;
	vector<Point> points;
	for (size_t idx = 0; idx < point_cnt; idx++)
	{
		Point point_on_line(face.x + ((int)idx * (face.width / (int)point_cnt)),
							face.y + ((int)idx * (face.height / (int)point_cnt)));
		points.push_back(point_on_line);
	}

	// Sometimes using the centroid fails even if the contour is the face, so going to use line from corner to
	// corner
	for (size_t idx = 0; idx < points.size(); idx++)
	{
		if (pointPolygonTest(contours, points[idx], false) >= 0)
		{
			return true;
		}
	}
	return false;
}

void GestureRecognition::FilterContours(vector<vector<Point>>& contours, Rect& face, Mat& image)
{
	for (vector<vector<Point>>::iterator contour_iterator = contours.begin(); contour_iterator != contours.end();)
	{
		if (fabs(contourArea(*contour_iterator)) < (face.area() / 3))
			contour_iterator = contours.erase(contour_iterator);
		else if (fabs(contourArea(*contour_iterator) > image.size().area() * 0.45)) // greater than 45 % image
			contour_iterator = contours.erase(contour_iterator);
		else
			++contour_iterator;
	}
	for (vector<vector<Point>>::iterator contour_iterator = contours.begin(); contour_iterator != contours.end();)
	{
		Point rightmost;
		rightmost.x = 0;
		Point leftmost;
		leftmost.x = INT_MAX;
		for (size_t idx = 0; idx < (*contour_iterator).size(); idx++)
		{
			if (rightmost.x < (*contour_iterator)[idx].x)
				rightmost = (*contour_iterator)[idx];
			if (leftmost.x > (*contour_iterator)[idx].x)
				leftmost = (*contour_iterator)[idx];
		}
		if (leftmost.x > face.x && leftmost.x < (face.x + face.width))
		{
			contour_iterator = contours.erase(contour_iterator);
		}
		else if (rightmost.x < (face.x + face.width) && rightmost.x > face.x)
		{
			contour_iterator = contours.erase(contour_iterator);
		}
		else if (leftmost.x < face.x && rightmost.x > face.x)
		{
			contour_iterator = contours.erase(contour_iterator);
		}
		else
		{
			++contour_iterator;
		}
	}
}

void GestureRecognition::ContourExamination(vector<vector<Point>>& contours,
											vector<Gesture>* gestures,
											Rect& face,
											Mat& image)
{
	FilterContours(contours, face, image);

	uint32_t hand_count = 0;
	for (size_t cont_idx = 0; cont_idx < contours.size(); cont_idx++)
	{
		// Check if any of the current gestures belong to the contour
		if (ContourIsFace(contours[cont_idx], face, image))
			continue;

		Gesture gesture;
		gesture.SetFace(face);
		gesture.SetTime(steady_clock::now());

		// An ever legible lambda function to find the top, left, and right point...
		Point hi_point = *min_element(
			contours[cont_idx].begin(), contours[cont_idx].end(), [](Point& lhs, Point& rhs) { return lhs.y < rhs.y; });
		if (hi_point.y > face.y + (face.height * 1.2))
			continue;
		Point l_point = *min_element(
			contours[cont_idx].begin(), contours[cont_idx].end(), [](Point& lhs, Point& rhs) { return lhs.x < rhs.x; });
		Point r_point = *max_element(
			contours[cont_idx].begin(), contours[cont_idx].end(), [](Point& lhs, Point& rhs) { return lhs.x < rhs.x; });
		gesture.SetHighPoint(hi_point);
		vector<Vec4i> defects = GetDefects(contours[cont_idx]);

		int highest_accepted = (hi_point.y > (face.height / 2)) ? hi_point.y - (face.height / 2) : 0;
		if (hi_point.y < highest_accepted)
			continue;

		// Should check vertical dist between highest point and defect to determine if is thumb or finger
		FingerDetection(contours[cont_idx], gesture, defects, image, face, hi_point, l_point, r_point);
		if (gesture.GetGesture()->fingers != FingerCount::Invalid)
		{
			int height_offset;
			if (gesture.GetGesture()->fingers == FingerCount::Fist)
				height_offset = (int)(face.height * 0.8) / 2;
			else
				height_offset = (int)(face.height * 0.8);
			Point cent((l_point.x + r_point.x) / 2, hi_point.y + height_offset);
			gesture.SetCentroid(cent);
		}
#ifdef DISPLAY_AND_ANNOTATE
		//	circle(image, info.centroid, (int)(height_offset / 2), COLOR_BLUE, 5, 8, 0);
#endif
		// Add gesture to gestures
		gestures->push_back(gesture);
	}
#ifdef DISPLAY_AND_ANNOTATE
	/*string contour_info = "";
	for (size_t gest_idx = 0; gest_idx < gestures->size(); gest_idx++)
	{
		if (gest_idx != 0 && gest_idx != (gestures->size() - 1))
		{
			contour_info += ", ";
		}
		else if ((gest_idx == (gestures->size() - 1)) && gestures->size() == 2)
		{
			contour_info += " and ";
		}
		else if (gest_idx == (gestures->size() - 1) && gestures->size() > 1)
		{
			contour_info += ", and ";
		}
		contour_info += (*gestures)[gest_idx].GestureToString();
	}
	Point pt((int)(image.size().width * .1), (int)(image.size().height * .9));
	putText(image, contour_info, pt, 1, 1, Scalar(255, 255, 0), 2);
	namedWindow("Contours", CV_WINDOW_NORMAL);
	imshow("Contours", image);
	waitKey(10);*/
#endif
}

vector<vector<Point>> GestureRecognition::GetContours(cv::Mat* image)
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

void GestureRecognition::DisplayPoints(Mat& image,
									   Point& start_pt,
									   Point& far_pt,
									   Point& end_pt,
									   CvScalar& colour_1,
									   CvScalar& colour_2,
									   CvScalar& colour_3)
{
#ifndef DISPLAY_AND_ANNOTATE
	line(image, far_pt, end_pt, colour_1, 2, CV_AA, 0);
	line(image, far_pt, start_pt, colour_1, 2, CV_AA, 0);
	line(image, start_pt, end_pt, colour_2, 2, CV_AA, 0);
	circle(image, far_pt, 5, colour_3, 2, 8, 0);
	circle(image, end_pt, 5, colour_3, 2, 8, 0);
	circle(image, start_pt, 5, colour_3, 2, 8, 0);
#else
	circle(image, far_pt, 5, colour_3, 2, 8, 0);
	circle(image, end_pt, 5, colour_3, 2, 8, 0);
	circle(image, start_pt, 5, colour_3, 2, 8, 0);
#endif
}

void GestureRecognition::FilterDefects(vector<Vec4i>& defects, Rect& face, vector<Point>& contours, Point& hi_point)
{
	for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();)
	{
		double defect_depth = (double)(defect_iterator->val[3] / 256.0f);
		if (defect_depth < (face.height * 0.15))
		{
			defect_iterator = defects.erase(defect_iterator);
			continue;
		}

		Point start_point(contours[defect_iterator->val[0]]);
		Point end_point(contours[defect_iterator->val[1]]);
		Point far_point(contours[defect_iterator->val[2]]);

		// Check that deep point is not too deep!
		if (far_point.y > (face.y + 1.75 * face.height))
		{
			// cout << "Removed based on location" << endl;
			defect_iterator = defects.erase(defect_iterator);
			continue;
		}
		// If start/end point's vertical position is quite low, skip over it
		if ((start_point.y > (hi_point.y + (face.height * 1.5))) || (end_point.y > hi_point.y + (face.height * 1.5)))
		{
			// cout << "Removed based on location" << endl;
			defect_iterator = defects.erase(defect_iterator);
			continue;
		}
		// If start point or end point is a decent bit below far point
		if ((start_point.y > (far_point.y + (face.height * 0.35))) ||
			(end_point.y > (far_point.y + (face.height * 0.35))))
		{
			// cout << "Removed based on location" << endl;
			defect_iterator = defects.erase(defect_iterator);
			continue;
		}
		defect_iterator++;
	}
}

void GestureRecognition::SkinExtraction(LPVOID lpParam)
{

	// Consider bilateral filter
	std::vector<Rect> faces;
	SkinInfo* skin_info = (SkinInfo*)lpParam;
	cuda::GpuMat* image = &(skin_info->image);
	Mat skin_extraction;
	image->download(skin_extraction);
	Mat skin_extraction_shrunk;
	// resize(skin_extraction, skin_extraction_shrunk, Size(640, 480), 0, 0, CV_INTER_AREA);

	Mat gray_mat;
	cvtColor(skin_extraction, gray_mat, CV_BGR2GRAY);
	equalizeHist(gray_mat, gray_mat);

	// Convert to GpuMat for cuda acceleration
	cv::cuda::GpuMat gray_gpumat;
	gray_gpumat.upload(gray_mat);

	// Detect all the faces in the image
	cv::cuda::GpuMat face_mat;

	// detectMultiscale uses cuda stream which is not thread safe and will crash
	lock_guard<mutex> lock(*(skin_info->cuda_stream));
	(*(skin_info->face_cascade))->detectMultiScale(gray_gpumat, gray_gpumat);
	(*(skin_info->face_cascade))->convert(gray_gpumat, faces);
	if (!faces.size())
	{
		// Should perform static skin detection in this case..... ?? ?? ? ? ? ? ? ? ? ? ? ??????
		return;
	}

	// Generate YCrCb representation of image
	Mat ycrcb_mat;
	cvtColor(skin_extraction, ycrcb_mat, CV_BGR2YCrCb);
	vector<Mat> ycrcb_split_init;
	split(ycrcb_mat, ycrcb_split_init);
	// normalize(ycrcb_mat, ycrcb_mat, 0, 255, NORM_MINMAX);

	// Generate vector of faces in image
	std::vector<Mat> face_ycrcb;
	vector<Rect> faces_shrunk;
	for (size_t face_idx = 0; face_idx < faces.size(); face_idx++)
	{
		if (faces[face_idx].area() < 100)
		{
			continue;
		}
		else if (face_idx > 0)
		{
			bool same = false;
			for (size_t face_idx2 = 0; face_idx2 < faces.size(); face_idx2++)
			{
				if (face_idx2 != face_idx)
				{
					if ((faces[face_idx2] & faces[face_idx]).area() > 0)
					{
						same = true;
						break;
					}
				}
			}
			if (same)
			{
				continue;
			}
		}
		Rect face_shrunk;
		face_shrunk.x = faces[face_idx].x + (int)(0.225 * faces[face_idx].width);
		face_shrunk.width = (int)(0.55 * faces[face_idx].width);
		face_shrunk.y = faces[face_idx].y; // -faces[face_idx].y * 0.125; // -0.10 * faces[face_idx].y;
		face_shrunk.height = (int)(faces[face_idx].height * 0.80);
		faces_shrunk.push_back(face_shrunk);
		face_ycrcb.push_back(ycrcb_mat(face_shrunk));
	}

	// Create back projection vectors
	std::vector<Mat> ycrcb_backproj_vec;
	bool accumulate = false;
	bool uniform = true;
	for (size_t ycrcb_idx = 0; ycrcb_idx < face_ycrcb.size(); ycrcb_idx++)
	{
		Mat ycrcb_hist, ycrcb_backproj;
		calcHist(&(face_ycrcb[ycrcb_idx]),
				 1,
				 ycrcb_channels,
				 Mat(),
				 ycrcb_hist,
				 1,
				 hist_size,
				 ycrcb_ranges,
				 uniform,
				 accumulate);

		calcBackProject(&(ycrcb_mat), 1, ycrcb_channels, ycrcb_hist, ycrcb_backproj, ycrcb_ranges, (double)1, true);
		ycrcb_backproj_vec.push_back(ycrcb_backproj);
	}

	if (faces_shrunk.size() != ycrcb_backproj_vec.size())
	{
		cout << "ERROR, SIZES ARE NOT EQUAL" << endl;
		return;
	}

	Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(4, 4));
	for (size_t idx = 0; idx < faces_shrunk.size(); idx++)
	{
#ifdef LOG_COUT
		// cout << "Pushing in one." << endl;
#endif
		Mat output_mat(ycrcb_backproj_vec[idx].size(), CV_8UC1);
		threshold(ycrcb_backproj_vec[idx],
				  output_mat,
				  (double)MIN_CHANCE(25),
				  255,
				  CV_THRESH_BINARY); // was 80 with below stuff

		// dilate(output_mat, output_mat, kernel);
		// dilate(output_mat, output_mat, kernel);
		// erode(output_mat, output_mat, kernel);
		// erode(output_mat, output_mat, kernel);
		FaceInfo in;
		skin_info->face_info.push_back(in);
		skin_info->face_info[idx].face_rect = faces_shrunk[idx];
		skin_info->face_info[idx].skin_mask = output_mat.clone();
	}
#ifdef TUNING_SKIN_DETECTION
	namedWindow("Skin detection", CV_WINDOW_NORMAL);
	imshow("Skin detection", skin_info->face_info[0].skin_mask); // ycrcb_backproj_vec[idx]);
#endif
	return;
}
