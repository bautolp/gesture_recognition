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
	SkinInfo skin_info;
	skin_info.face_cascade = &m_face_cascade;
	skin_info.image = image;
	skin_info.cuda_stream = m_cuda_stream;
	std::thread skin_handle(SkinExtraction, &skin_info);
#ifdef DISPLAY_AND_ANNOTATE
	Mat display_img(image.size(), image.type());
	image.download(display_img);
#else
	Mat display_img;
#endif
	skin_handle.join();

	// Should profile this time and if it's high, thread
	vector<Gesture> gestures;
	for (size_t index = 0; index < skin_info.face_info.size(); index++)
	{
#ifdef DISPLAY_AND_ANNOTATE
		namedWindow("Mask" + to_string(index), CV_WINDOW_NORMAL);
		imshow("Mask" + to_string(index), skin_info.face_info[index].skin_mask);
		waitKey(10);
#endif
		vector<vector<Point>> contours = GetContours(&skin_info.face_info[index].skin_mask);
		ContourExamination(contours, &gestures, skin_info.face_info[index].face_rect, display_img);
	}
	for (size_t gest_idx = 0; gest_idx < gestures.size(); gest_idx++)
	{
		gestures[gest_idx].SetTime(timestamp);
	}
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

bool GestureRecognition::AddIfNotInVector(vector<Point>* points, Point& point, int face_width)
{
	for (size_t point_idx = 0; point_idx < points->size(); point_idx++)
	{
		if (point.x >= ((*points)[point_idx].x - face_width * 0.15) &&
			point.x <= ((*points)[point_idx].x + face_width * 0.15) &&
			point.y >= ((*points)[point_idx].y - face_width * 0.15) &&
			point.y <= ((*points)[point_idx].y + face_width * 0.15))
		{
			return false;
		}
	}
	points->push_back(point);
	return true;
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
	// Find contour points that are near hi point vertically, but further left or further right
	int width_for_check = (int)(face.height * 0.2);
	for (size_t idx = 0; idx < contours.size(); idx++)
	{
		if (X_WITHIN_Y_AMT(contours[idx].y, hi_point.y, width_for_check))
		{
			if (r_near_hi.x < contours[idx].x)
				r_near_hi = contours[idx];
			if (l_near_hi.x > contours[idx].x)
				l_near_hi = contours[idx];
			if (contours[idx].y < (hi_point.y + face.height * 1.5))
			{
				if (r_above_wrist.x > contours[idx].x)
					r_above_wrist = contours[idx];
				if (l_above_wrist.x < contours[idx].x)
					l_above_wrist = contours[idx];
			}
		}
	}

	FilterDefects(defects, face, contours, hi_point);
	for (vector<Vec4i>::iterator defect_iterator = defects.begin(); defect_iterator != defects.end();)
	{
		// TODO make filter defects a prelimiary parse which returns whether left or right so that can be set once
		// TODO make a isthumb function for the defect
		// Todo make an is single finger function for the defect
		// todo make an is multiple fingers function for the defect
		// TODO reduce indentation depth if the above does not do that (which it should)
		Point start_point(contours[defect_iterator->val[0]]);
		Point end_point(contours[defect_iterator->val[1]]);
		Point far_point(contours[defect_iterator->val[2]]);

		// Get leftmost and rightmost points... Need to check their distance to far point
		Point left(INT_MAX, 0);
		Point right(0, 0);
		right = (start_point.x > right.x) ? start_point : right;
		right = (end_point.x > right.x) ? end_point : right;
		left = (start_point.x < left.x) ? start_point : left;
		left = (end_point.x < left.x) ? end_point : left;
		bool head_to_right = (face.x > hi_point.x);

		// Face is on the right of the hand
		if (head_to_right)
		{
			gesture.SetSide(Right);
			if ((right.x >= (r_near_hi.x + face.width * 0.125)) && (right.x >= (r_above_wrist.x - face.width * 0.1)))
			{
				// Add check that it points towards head
				if (gesture.GetThumb() != Out)
					gesture.SetThumb(Out);
				else
				{
					gesture.SetFingerCount(FingerCount::Invalid);
					cout << "THUMB DETECTED TWICE, SETTING INVALID (right hand (head on right))" << endl;
					break;
				}
				DisplayPoints(image, start_point, far_point, end_point, COLOR_YELLOW, COLOR_AQUA, COLOR_PURPLE);

				defect_iterator = defects.erase(defect_iterator);
				continue;
			}
			else
			{
				/*
				cout << "Right: " << right << endl;
				cout << "Right Above wrist: " << r_above_wrist << endl;
				cout << "Right near hi: " << r_near_hi << endl;
				cout << "Face width: " << face.width * 0.125 << endl;
				*/
			}
		}
		// Face is on left
		else
		{
			gesture.SetSide(Left);
			if ((left.x <= (l_near_hi.x - face.width * 0.125)) && (left.x <= (l_above_wrist.x + face.width * 0.1)))
			{
				// Add check that it points towards head
				if (gesture.GetThumb() != Out)
					gesture.SetThumb(Out);
				else
				{
					gesture.SetFingerCount(FingerCount::Invalid);
					cout << "THUMB DETECTED TWICE, SETTING INVALID (left hand (head on left))" << endl;
					break;
				}
				DisplayPoints(image, start_point, far_point, end_point, COLOR_PINK, COLOR_BLUE, COLOR_GREEN);

				defect_iterator = defects.erase(defect_iterator);
				continue;
			}
			else
			{
				/*
				cout << "Left: " << left << endl;
				cout << "Left Above wrist: " << l_above_wrist << endl;
				cout << "Left near hi: " << l_near_hi << endl;
				cout << "Left width: " << face.width * 0.125 << endl;
				*/
			}
		}

		// If vertical distance between start point and end point is about a finger tip length, we assume it is between
		// two fingers
		if (X_WITHIN_Y_AMT(start_point.y, end_point.y, (face.height * 0.25)))
		{
			// Have we counted these finger tips yet?
			finger_count =
				(AddIfNotInVector(&counted_points, start_point, face.width)) ? finger_count + 1 : finger_count;
			finger_count = (AddIfNotInVector(&counted_points, end_point, face.width)) ? finger_count + 1 : finger_count;
			DisplayPoints(image, start_point, far_point, end_point, COLOR_DARK_GREEN, COLOR_DARK_BLUE, COLOR_DARK_RED);

			defect_iterator = defects.erase(defect_iterator);
			continue;
		}
		// If vertical distance between start and end point is large and it is not thumb, it is likely single finger
		else
		{
			// If head is on the right, check if we're looking at the leftmost point, if so, we are likely looking at
			// the pinky in extended gesture
			// ALso check for rational height
			if (head_to_right && (left.x <= (l_above_wrist.x + face.width * 0.1)) &&
				(left.y <= (hi_point.y + face.height * 0.5)))
			{
				// This is probably the pinky... If the other point is above it, check if it isn't added and add it too!
				if (AddIfNotInVector(&counted_points, left, face.width))
				{
					finger_count++;
					DisplayPoints(image,
								  start_point,
								  far_point,
								  end_point,
								  COLOR_DARK_YELLOW,
								  COLOR_DARK_PURPLE,
								  COLOR_DARK_TEAL);
				}
				if (left.y > right.y)
				{
					if (AddIfNotInVector(&counted_points, right, face.width))
					{
						finger_count++;
					}
				}
			}
			// If head is on the left, check if we're looking at the rightmost point, if so, we are likely looking at
			// the pinky in extended gesture
			// Also check that it has a rational height
			else if (!head_to_right && (right.x >= (r_above_wrist.x - face.width * 0.1)) &&
					 (right.y <= (hi_point.y + face.height * 0.5)))
			{
				// This is probably the pinky...
				if (AddIfNotInVector(&counted_points, right, face.width))
				{
					finger_count++;
					DisplayPoints(image, start_point, far_point, end_point, COLOR_AQUA, COLOR_DARK_YELLOW, COLOR_PINK);
				}
				// If the other point is above it, check if it isn't added and add it too!
				if (right.y > left.y)
				{
					if (AddIfNotInVector(&counted_points, left, face.width))
					{
						finger_count++;
					}
				}
			}
			// Generic single finger handler
			else
			{
				Point higher = start_point;
				if (end_point.y < higher.y)
					higher = end_point;
				if (AddIfNotInVector(&counted_points, higher, face.width))
				{
					finger_count++;
					DisplayPoints(
						image, start_point, far_point, end_point, COLOR_GREEN, COLOR_DARK_GREEN, COLOR_DARK_PURPLE);
				}
			}
			defect_iterator = defects.erase(defect_iterator);
			continue;
		}

		defect_iterator++;
	}

	if (!gesture.GetGesture()->fingers == FingerCount::Invalid)
		DetermineGesture(&gesture, finger_count);
}

bool GestureRecognition::ContourIsFace(std::vector<cv::Point>& contours, cv::Rect& face, cv::Mat& image)
{
	Point centroid;
	centroid.x = face.x + (face.width / 2);
	centroid.y = face.y + (face.height / 2);
#ifdef DISPLAY_AND_ANNOTATE
	rectangle(image, face, Scalar(255, 255, 0), 3, 8, 0);
#endif
	if (pointPolygonTest(contours, centroid, false) >= 0)
	{
		return true;
	}
	return false;
}

void GestureRecognition::FilterContours(vector<vector<Point>>& contours, Rect& face, Mat& image)
{
	for (vector<vector<Point>>::iterator contour_iterator = contours.begin(); contour_iterator != contours.end();)
	{
		if (fabs(contourArea(*contour_iterator)) < (face.area() / 2))
			contour_iterator = contours.erase(contour_iterator);
		else if (fabs(contourArea(*contour_iterator) > image.size().area() * 0.45)) // greater than 45 % image
			contour_iterator = contours.erase(contour_iterator);
		else
			++contour_iterator;
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

		vector<Vec4i> defects = GetDefects(contours[cont_idx]);

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
	string contour_info = "";
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
	waitKey(10);
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
#ifdef DISPLAY_AND_ANNOTATE
	line(image, far_pt, end_pt, colour_1, 2, CV_AA, 0);
	line(image, far_pt, start_pt, colour_1, 2, CV_AA, 0);
	line(image, start_pt, end_pt, colour_2, 2, CV_AA, 0);
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
		if (defect_depth < (face.height * 0.10))
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
		if ((start_point.y > (hi_point.y + face.height * 1.2)) || (end_point.y > (hi_point.y + face.height * 1.2)))
		{
			// cout << "Removed based on location" << endl;
			defect_iterator = defects.erase(defect_iterator);
			continue;
		}
		defect_iterator++;
	}
}

DWORD WINAPI GestureRecognition::SkinExtraction(LPVOID lpParam)
{
	std::vector<Rect> faces;
	SkinInfo* skin_info = (SkinInfo*)lpParam;
	cuda::GpuMat* image = &(skin_info->image);
	Mat skin_extraction;
	image->download(skin_extraction);
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

	// detectMultiscale uses cuda stream which is not thread safe and will crash
	lock_guard<mutex> lock(*(skin_info->cuda_stream));
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
	// normalize(ycrcb_mat, ycrcb_mat, 0, 255, NORM_MINMAX);

	// Generate vector of faces in image
	std::vector<Mat> face_ycrcb;
	vector<Rect> faces_shrunk;
	for (size_t face_idx = 0; face_idx < faces.size(); face_idx++)
	{
		Rect face_shrunk;
		face_shrunk.x = faces[face_idx].x + (int)(0.225 * faces[face_idx].width);
		face_shrunk.width = (int)(0.55 * faces[face_idx].width);
		face_shrunk.y = faces[face_idx].y; // -faces[face_idx].y * 0.125; // -0.10 * faces[face_idx].y;
		face_shrunk.height = (int)(faces[face_idx].height * 0.80);
		faces_shrunk.push_back(face_shrunk);
		face_ycrcb.push_back(ycrcb_mat(face_shrunk));
	}

	// General histogram parameters... Use all 3 channels, with max size bins
	int ycrcb_channels[] = {1, 2};
	int channel_count = 2;
	int image_count = 1;
	int hist_size[] = {5, 10}; // 122, 122
	bool accumulate = false;
	bool uniform = true;

	// YCrCb information, allow max ranges
	float yranges[] = {0, 255};
	float crranges[] = {130, 180}; // {130, 180}; // 138 to 173
	float cbranges[] = {60, 140};  //  {60, 140}; // 67 to 133
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
		calcBackProject(
			&ycrcb_mat, image_count, ycrcb_channels, ycrcb_hist, ycrcb_backproj, ycrcb_ranges, (double)1, true);
		ycrcb_backproj_vec.push_back(ycrcb_backproj);
	}

	if (faces_shrunk.size() != ycrcb_backproj_vec.size())
	{
		cout << "ERROR, SIZES ARE NOT EQUAL" << endl;
		return 1;
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

		Mat final_output(ycrcb_backproj_vec[idx].size(), CV_8UC1);
		dilate(output_mat, output_mat, kernel);
		erode(output_mat, output_mat, kernel);
		erode(output_mat, output_mat, kernel);
		FaceInfo in;
		skin_info->face_info.push_back(in);
		skin_info->face_info[idx].face_rect = faces_shrunk[idx];
		skin_info->face_info[idx].skin_mask = output_mat.clone();
	}

#ifdef LOG_COUT
	// cout << "Updated output image..." << endl;
#endif

	return 0;
}

void PrintHello(void*)
{
	cout << endl;
	cout << endl;
	cout << endl;
	cout << endl;
	cout << endl;
	cout << "HELLO WORLD!";
	cout << endl;
	cout << endl;
	cout << endl;
	cout << endl;
	cout << endl;
	cout << endl;
}

// Entry point
int main()
{
	VideoCapture cap(0); // open the default camera
	string filename = "cellphone_video2.mp4";
	VideoCapture file_cap(filename);
	Mat frame;

	if (!file_cap.isOpened())
		exception e("Error when reading steam_avi");

	if (!cap.isOpened())
	{
		exception e("Failed to open camera.");
	}
	uint64_t count = 0;
	Mat image;

	for (uint32_t i = 0; i < 15; i++)
	{
		if (!cap.read(image))
		{
			exception e("Failed to grab first image.");
		}
	}

	cout << image.size() << endl;
	cuda::GpuMat gpu_image;
	gpu_image.upload(image);

	GestureRecognition gesture_recognition;
	while (1)
	{
		try
		{
			// read img to pass to class
			vector<string> images_to_process;
			/*
			int frame_width = file_cap.get(CV_CAP_PROP_FRAME_WIDTH);
			int frame_height = file_cap.get(CV_CAP_PROP_FRAME_HEIGHT);
			VideoWriter video("out.avi", CV_FOURCC('D', 'I', 'V', 'X'), 10, Size(frame_width, frame_height), true);
			VideoWriter bw_video(
				"bw_out.avi", CV_FOURCC('D', 'I', 'V', 'X'), 10, Size(frame_width, frame_height), true);*/
			steady_clock::time_point start = steady_clock::now();
			uint32_t frame_count = 0;
			GestureTracker tracker;
			GestureInfo start_gest;
			start_gest.fingers = FourFingers;
			start_gest.thumb = Out;
			GestureInfo end_gest;
			end_gest.fingers = TwoFingers;
			end_gest.thumb = Out;

			/*
			Go through contours, simplify it, and make sure it is setting
			all
			settable
			method
			because
			some
			might
			not
			be
			being
			done
			Look for FIXME and TODO and do them finally
			*/
			tracker.RegisterCallback(start_gest, end_gest, PrintHello);
			function<void(void*)> fun = PrintHello;
			fun(NULL);
			for (;;)
			{
				frame_count++;
				file_cap >> frame;
				if (frame.empty())
					break;
				gpu_image.upload(frame);
				std::vector<Gesture> gestures;
				gestures.clear();
				gestures = gesture_recognition.GetGestureInfo(gpu_image, steady_clock::now());
				tracker.AddGesture(gestures);
			}
			steady_clock::time_point end = steady_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(end - start);

			std::cout << "It took " << time_span.count() << " seconds to process " << frame_count << " frames.";
			break;
		}
		catch (cv::Exception& e)
		{
			cout << "OpenCV exception: " << e.what();
			break;
		}
		catch (exception& e)
		{
			cout << "Exception: " << e.what();
			break;
		}
	}

	cout << "Exception caused ending of program." << endl;
	return 0;
}
