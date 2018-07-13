#include "GestureRecognition.h"
#include "GestureExceptions.h"
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

// High tolerance ranges used, just as a preliminary filter of skin, will not
// block much non skin
const int y_max_max = 255;
const int y_min_max = 255;
const int cb_max_max = 255;
const int cb_min_max = 255;
const int cr_max_max = 255;
const int cr_min_max = 255;

int y_max = y_max_max;
int y_min = 0;
int cb_max = cb_max_max;
int cb_min = 0;
int cr_max = cr_max_max;
int cr_min = 0;

const int h_max_max = 189;
const int h_min_max = 189;
const int s_max_max = 255;
const int s_min_max = 255;
const int v_max_max = 255;
const int v_min_max = 255;

int h_max = h_max_max;
int h_min = 0;
int s_max = s_max_max;
int s_min = 0;
int v_max = v_max_max;
int v_min = 0;

const int max_dim = CV_MAX_DIM;

int bins = max_dim;

#define Y_MIN 0
#define Y_MAX 255
#define Cb_MIN 0
#define Cb_MAX 255
#define Cr_MIN 0
#define Cr_MAX 255

#define H_MIN 0
#define H_MAX 189
#define S_MIN 0
#define S_MAX 256
#define V_MIN 0
#define V_MAX 256

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

#define EYE_CASCADE_PATH                                                       \
  "C:\\Users\\lyndo\\Developer\\OpenCV_"                                       \
  "cuda\\install\\etc\\haarcascades\\haarcascade_eye.xml"

#define FACE_CASCADE_PATH                                                      \
  "C:\\Users\\lyndo\\Developer\\OpenCV_"                                       \
  "cuda\\install\\etc\\haarcascades\\haarcascade_"                             \
  "frontalface_alt.xml"

using namespace cv;
using namespace cuda;
using namespace std;

// Returns the square of the euclidean distance between 2 points.
inline double dist(Point a, Point b) {
  double x = (double)(a.x - b.x);
  double y = (double)(a.y - b.y);
  return (x * x) + (y * y);
}

// Returns the angle that the line from b to a makes with the line from b to c
inline uint32_t AngleABC(Point a, Point b, Point c) {
  Point ab = {b.x - a.x, b.y - a.y};
  Point cb = {b.x - c.x, b.y - c.y};

  float dot = (float)(ab.x * cb.x + ab.y * cb.y);   // dot product
  float cross = (float)(ab.x * cb.y - ab.y * cb.x); // cross product

  float alpha = atan2(cross, dot);

  return (uint32_t)abs(floor(alpha * 180. / pi + 0.5));
}

void on_trackbar(int, void *) {}

GestureRecognition::GestureRecognition(uint64_t image_payload_size) {
  m_runtime_options.background_image_count = BACKGROUND_QUEUE_SIZE;
  m_runtime_options.dynamic_background_subtraction = true;
  m_runtime_options.image_payload_size = image_payload_size;

  InitializeBackgroundQueue(image_payload_size,
                            m_runtime_options.background_image_count);
  cv::cuda::GpuMat new_mat(Size(150, 150), CV_16UC1);

  try {

    cout << FACE_CASCADE_PATH << endl;
    cout << EYE_CASCADE_PATH << endl;
    m_face_cascade = cv::cuda::CascadeClassifier::create(FACE_CASCADE_PATH);
    m_eye_cascade = cv::cuda::CascadeClassifier::create(EYE_CASCADE_PATH);

  } catch (cv::Exception &e) {
    cout << e.what();
    cout << e.what();
  }
}

GestureRecognition::~GestureRecognition() {} // cvDestroyAllWindows(); }

bool GestureRecognition::DetectPalm(cv::Point *location,
                                    cv::cuda::GpuMat *image) {
  // BackgroundSubtraction(image);
  Mat skin_filter = SkinExtraction(image);
  // SkinExtraction(image, &skin_filter);
  // if (!ContourExtraction(image)) {
  //  return false;
  //}
  return true;
}

void GestureRecognition::BackgroundSubtraction(cv::cuda::GpuMat *image) {
  // Call background subtraction class?
}

void GestureRecognition::PushBackgroundImage(cv::cuda::GpuMat *image) {
  m_background_image_queue.Push(image);
}

// This should probably return information about the palm, maybe extreme
// points of contour extraction
bool GestureRecognition::ContourExtraction(cv::cuda::GpuMat *image) {
  int actual_contours = 0;

  imshow("Original image", *image);

  cv::cuda::GpuMat ycrcb;
  blur(*image, *image, Size(2, 2));
  cvtColor(*image, ycrcb, CV_BGR2YCrCb);
  cv::cuda::GpuMat skin_mask;
  inRange(ycrcb, Scalar(Y_MIN, Cr_MIN - 10, Cb_MIN),
          Scalar(Y_MAX, Cr_MAX, Cb_MAX + 10), skin_mask);

  cv::cuda::GpuMat multiplied;
  bitwise_and(*image, Scalar(255, 255, 255), multiplied, skin_mask);

  imshow("Skin mask", skin_mask);
  namedWindow("skin color", 1);
  imshow("skin color", multiplied);
  waitKey(-1);
  imwrite("ycrcb_img_multiplied.png", multiplied);
  // Extract contours of image

  std::chrono::time_point<std::chrono::system_clock> before_contours =
      std::chrono::system_clock::now();

  vector<vector<Point>> contours;
  if (FindContours(&skin_mask, &contours) <= 0) {
    return false; // Could not find anything. Maybe return something else
                  // after
  }

  std::chrono::time_point<std::chrono::system_clock> after_contours =
      std::chrono::system_clock::now();

  // Remove contours that are too small
  for (vector<vector<Point>>::iterator contour_iterator = contours.begin();
       contour_iterator != contours.end();) {
    if (fabs(contourArea(*contour_iterator)) < MINIMUM_CONTOUR_AREA)
      contour_iterator = contours.erase(contour_iterator);
    else
      ++contour_iterator;
  }

  std::chrono::time_point<std::chrono::system_clock> after_areas =
      std::chrono::system_clock::now();

  uint32_t hand_count = 0;
  for (size_t contour_idx = 0; contour_idx < contours.size(); contour_idx++) {
    vector<vector<int>> hull(1);
    vector<vector<Vec4i>> defects(1);
    convexHull(Mat(contours[contour_idx]), hull[0], false);
    convexityDefects(contours[contour_idx], hull[0], defects[0]);

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
    for (vector<Vec4i>::iterator defect_iterator = defects[0].begin();
         defect_iterator != defects[0].end();) {
      double defect_depth = (double)((*defect_iterator).val[3] / 256.0f);
      if (defect_depth < MINIMUM_DEFECT_DEPTH) {
        defect_iterator = defects[0].erase(defect_iterator);
        continue;
      }

      int start_index = (*defect_iterator).val[0];
      Point start_point(contours[contour_idx][start_index]);
      int end_index = (*defect_iterator).val[1];
      Point end_point(contours[contour_idx][end_index]);
      int far_index = (*defect_iterator).val[2];
      Point far_point(contours[contour_idx][far_index]);

      if (far_point.x > rightmost_defect.x) {
        rightmost_defect.x = far_point.x;
        rightmost_defect.y = far_point.y;
      }
      if (far_point.x < leftmost_defect.x) {
        leftmost_defect.x = far_point.x;
        leftmost_defect.y = far_point.y;
      }
      if (far_point.y > highest_defect.y) {
        highest_defect.x = far_point.x;
        highest_defect.y = far_point.y;
      }
      if (far_point.y < lowest_defect.y) {
        lowest_defect.x = far_point.x;
        lowest_defect.y = far_point.y;
      }

      uint32_t angle = AngleABC(start_point, far_point, end_point);
      if (angle < ANGLE_BETWEEN_FINGERS) {
        fingers_found++;
        average_finger_separation += (int)dist(start_point, end_point);
        average_finger_length +=
            (int)(dist(far_point, start_point) + dist(far_point, end_point)) /
            2;
#ifdef _DEBUG
        line(*image, start_point, far_point, COLOR_RED, 1, CV_AA, 0);
        circle(*image, far_point, 5, COLOR_TEAL, 2, 8, 0);
        circle(*image, start_point, 5, COLOR_PURPLE, 2, 8, 0);
        line(*image, start_point, end_point, COLOR_BLUE, 1, CV_AA, 0);
        line(*image, far_point, end_point, COLOR_GREEN, 1, CV_AA, 0);
#endif

        defect_iterator = defects[0].erase(defect_iterator);
        continue;
      }
      defect_iterator++;
    }

    if (fingers_found) {
      hand_count++;
    } else {
      continue;
    }

    // cout << "Finger count: " << fingers_found << endl;
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
    for (size_t palm_dist_idx = 1; palm_dist_idx < palm_distances.size();
         palm_dist_idx++) {
      if (palm_distances[palm_dist_idx] > farthest) {
        farthest = palm_distances[palm_dist_idx];
      }
      if (palm_distances[palm_dist_idx] < shortest) {
        shortest = palm_distances[palm_dist_idx];
      }
    }
    circle(*image, center_of_palm, (int)sqrt(shortest), COLOR_BLUE, 2, 8, 0);

    namedWindow("palm", 1);
    imshow("palm", *image);
    waitKey(-1);

    for (vector<Vec4i>::iterator defect_iterator = defects[0].begin();
         defect_iterator != defects[0].end(); defect_iterator++) {
      int start_index = (*defect_iterator).val[0];
      Point start_point(contours[contour_idx][start_index]);
      int end_index = (*defect_iterator).val[1];
      Point end_point(contours[contour_idx][end_index]);
      int far_index = (*defect_iterator).val[2];
      Point far_point(contours[contour_idx][far_index]);

      uint32_t angle = AngleABC(start_point, far_point, end_point);
      if (angle > ANGLE_BETWEEN_FINGERS) {
        // Could be a thumb, check the length.
        int length =
            (int)(dist(far_point, end_point) + dist(far_point, start_point)) /
            2;
        if (length > (2 * average_finger_length)) {
#ifdef _DEBUG
          line(*image, start_point, far_point, COLOR_DARK_RED, 1, CV_AA, 0);
          circle(*image, far_point, 5, COLOR_DARK_TEAL, 2, 8, 0);
          circle(*image, start_point, 5, COLOR_DARK_PURPLE, 2, 8, 0);
          line(*image, start_point, end_point, COLOR_DARK_BLUE, 1, CV_AA, 0);
          line(*image, far_point, end_point, COLOR_DARK_GREEN, 1, CV_AA, 0);
#endif
        } else {
          thumb_found = true;
// cout << "Found thumb " << endl;
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
}

void GestureRecognition::BlurImage(cv::cuda::GpuMat *image) {
  blur(*image, *image, BLUR_SIZE);
#ifdef _DEBUG
  namedWindow("Blur", 1);
  imshow("Blur", *image);
  waitKey(0);
#endif
}

void GestureRecognition::ThresholdImage(cv::cuda::GpuMat *image) {
  threshold(*image, *image, 0, 255, THRESH_BINARY + THRESH_OTSU);
#ifdef _DEBUG
  namedWindow("Threshold", 1);
  imshow("Threshold", *image);
  waitKey(0);
#endif
}

void GestureRecognition::GenerateTrackerbars() {
  namedWindow("Bins", 1);
  createTrackbar("Bins", "Linear Blend", &bins, max_dim, on_trackbar);
  waitKey(1000);
  namedWindow("Y_min", 1);
  createTrackbar("Y_min", "Linear Blend", &y_min, y_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("Y_max", 1);
  createTrackbar("Y_max", "Linear Blend", &y_max, y_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("Cr_min", 1);
  createTrackbar("Cr_min", "Linear Blend", &cr_min, cr_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("Cr_max", 1);
  createTrackbar("Cr_max", "Linear Blend", &cr_max, cr_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("Cb_min", 1);
  createTrackbar("Cb_min", "Linear Blend", &cb_min, cb_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("Cb_max", 1);
  createTrackbar("Cb_min", "Linear Blend", &cb_min, cb_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("H_min", 1);
  createTrackbar("H_min", "Linear Blend", &h_min, h_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("H_max", 1);
  createTrackbar("H_max", "Linear Blend", &h_max, h_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("S_min", 1);
  createTrackbar("S_min", "Linear Blend", &s_min, s_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("S_max", 1);
  createTrackbar("S_max", "Linear Blend", &s_max, s_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("V_min", 1);
  createTrackbar("V_min", "Linear Blend", &v_min, v_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("V_max", 1);
  createTrackbar("V_max", "Linear Blend", &v_max, v_max_max, on_trackbar);
  waitKey(1000);
  namedWindow("probability", 1);
  createTrackbar("probability", "Linear Blend", &min_probability, 255,
                 on_trackbar);
  waitKey(1000);
}

size_t GestureRecognition::FindContours(cv::cuda::GpuMat *image,
                                        vector<vector<Point>> *contours) {
  // Find contours
  vector<Vec4i> hierarchy;
  findContours(*image, *contours, hierarchy, CV_RETR_TREE,
               CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

  return hierarchy.size();
}

Mat GestureRecognition::SkinExtraction(cv::cuda::GpuMat *image) {
  Mat skin_extraction;
  (*image).download(skin_extraction);

  // Create gray scale image and increase contrast w/ histogram
  cv::cuda::GpuMat gray; //, img_copy;
  namedWindow("Original", WINDOW_NORMAL);
  imshow("Original", skin_extraction);

  Mat gray_mat;
  cvtColor(skin_extraction, gray_mat, CV_BGR2GRAY);
  equalizeHist(gray_mat, gray_mat);
  // equalizeHist(skin_extraction, skin_extraction);
  gray.upload(gray_mat);

  // Detect all the faces in the image
  cv::cuda::GpuMat face_mat;
  std::vector<Rect> faces;
  m_face_cascade->detectMultiScale(gray, face_mat);
  m_face_cascade->convert(face_mat, faces);

  waitKey(1000);
  // Generate HSV and YCrCb representation of image
  Mat hsv_mat;
  Mat ycrcb_mat;
  cvtColor(skin_extraction, ycrcb_mat, CV_BGR2YCrCb);
  cvtColor(skin_extraction, hsv_mat, CV_BGR2HSV);

  // Generate vector of faces in image
  std::vector<Mat> eyeless_faces_hsv;
  std::vector<Mat> eyeless_faces_ycrcb;
  for (size_t face_idx = 0; face_idx < faces.size(); face_idx++) {
    Rect face_shrunk;
    face_shrunk.x = faces[face_idx].x + 0.10 * faces[face_idx].x;
    face_shrunk.width = faces[face_idx].width - 0.45 * faces[face_idx].width;
    face_shrunk.y = faces[face_idx].y; // -0.10 * faces[face_idx].y;
    face_shrunk.height = faces[face_idx].height - 0.10 * faces[face_idx].height;
    eyeless_faces_hsv.push_back(hsv_mat(face_shrunk));
    eyeless_faces_ycrcb.push_back(ycrcb_mat(face_shrunk));
    namedWindow("Eyelesshsv" + to_string(face_idx), WINDOW_NORMAL);
    waitKey(100);
    namedWindow("Eyelessycrcb" + to_string(face_idx), WINDOW_NORMAL);
    waitKey(100);
    imshow("Eyelesshsv" + to_string(face_idx), eyeless_faces_hsv[face_idx]);
    imshow("Eyelessycrcb" + to_string(face_idx), eyeless_faces_ycrcb[face_idx]);
  }

  if (!faces.size()) {
    // Will cause crash otherwise...... Need to probably throw here and catch to
    // skip other shit
    cout << "Probably about to shit.";
    return Mat();
  }

  // General histogram parameters... Use all 3 channels, with max size bins
  int channels[] = {0, 1, 2};
  int channel_count = 3;
  int image_count = 1;
  int bins_cnt = 3;
  int hist_size[] = {bins, bins, bins};
  bool accumulate = false;
  bool uniform = true;

  // HSV information, allow max ranges
  float hranges[] = {0, 25};
  float sranges[] = {10, 150};
  float vranges[] = {V_MIN, V_MAX};
  const float *hsv_ranges[] = {hranges, sranges};

  // YCrCb information, allow max ranges
  float yranges[] = {0, 255};
  float crranges[] = {0, 255};
  float cbranges[] = {0, 255};
  const float *ycrcb_ranges[] = {yranges, crranges, cbranges};

  // Create back projection vectors
  std::vector<Mat> hsv_backproj_vec, ycrcb_backproj_vec;
  for (size_t hsv_idx = 0; hsv_idx < eyeless_faces_hsv.size(); hsv_idx++) {
    // Generate histogram and normalize it

    /*Consider removing value from HSV, it might caught light sensitivity*/
    Mat hsv_hist;
    calcHist(&(eyeless_faces_hsv[hsv_idx]), image_count, channels, Mat(),
             hsv_hist, 2, hist_size, hsv_ranges, uniform, accumulate);
    // normalize(hsv_hist, hsv_hist, 0, 255, NORM_MINMAX, -1, Mat());

    // Get back projection
    Mat hsv_backproj;
    calcBackProject(&hsv_mat, image_count, channels, hsv_hist, hsv_backproj,
                    hsv_ranges, 1, true);
    namedWindow("Backproj_hsv_" + to_string(hsv_idx), WINDOW_NORMAL);
    imshow("Backproj_hsv_" + to_string(hsv_idx), hsv_backproj);
    hsv_backproj_vec.push_back(hsv_backproj);
    waitKey(100);
  }
  for (size_t ycrcb_idx = 0; ycrcb_idx < eyeless_faces_ycrcb.size();
       ycrcb_idx++) {
    // Generate histogram and normalize it
    Mat ycrcb_hist;
    calcHist(&(eyeless_faces_ycrcb[ycrcb_idx]), image_count, channels, Mat(),
             ycrcb_hist, channel_count, hist_size, ycrcb_ranges, uniform,
             accumulate);

    // Get back projection
    Mat ycrcb_backproj;
    calcBackProject(&ycrcb_mat, image_count, channels, ycrcb_hist,
                    ycrcb_backproj, ycrcb_ranges, 1, true);
    namedWindow("Backproj_ycrcb_" + to_string(ycrcb_idx), WINDOW_NORMAL);
    imshow("Backproj_ycrcb_" + to_string(ycrcb_idx), ycrcb_backproj);
    ycrcb_backproj_vec.push_back(ycrcb_backproj);
    waitKey(100);
  }

  // Get highest change out of all mats that pixel is skin
  Mat highest_probability(skin_extraction.size(), CV_8UC1); // Scalar::all(0));
  highest_probability.setTo(0);
  // Could copy first hsv thing into highest probability, rather than all 0's,
  // would slightly improve performance

  for (size_t hsv_idx = 0; hsv_idx < hsv_backproj_vec.size(); hsv_idx++) {
    MatIterator_<unsigned char> pixel_iter, hsv_iter, end;
    for (pixel_iter = highest_probability.begin<unsigned char>(),
        hsv_iter = hsv_backproj_vec[hsv_idx].begin<unsigned char>(),
        end = highest_probability.end<unsigned char>();
         pixel_iter != end; ++pixel_iter, ++hsv_iter) {
      if (*pixel_iter < *hsv_iter) {
        *pixel_iter = *hsv_iter;
      }
    }
  }
  for (size_t ycrcb_idx = 0; ycrcb_idx < ycrcb_backproj_vec.size();
       ycrcb_idx++) {
    MatIterator_<unsigned char> pixel_iter, ycrcb_iter, end;
    for (pixel_iter = highest_probability.begin<unsigned char>(),
        ycrcb_iter = ycrcb_backproj_vec[ycrcb_idx].begin<unsigned char>(),
        end = highest_probability.end<unsigned char>();
         pixel_iter != end; ++pixel_iter, ++ycrcb_iter) {
      if (*pixel_iter < *ycrcb_iter) {
        *pixel_iter = *ycrcb_iter;
      }
    }
  }

  unsigned char pxl = 0;
  uint64_t above_threshold_count = 0;
  uint64_t below_threshold_count = 0;
  uint64_t chance_100 = 0;
  uint64_t chance_95 = 0;
  uint64_t chance_90 = 0;
  uint64_t chance_85 = 0;
  uint64_t chance_80 = 0;
  uint64_t chance_75 = 0;
  uint64_t chance_70 = 0;
  uint64_t chance_65 = 0;
  uint64_t chance_60 = 0;
  uint64_t chance_55 = 0;
  uint64_t chance_50 = 0;
  uint64_t chance_45 = 0;
  uint64_t chance_40 = 0;
  uint64_t chance_35 = 0;
  uint64_t chance_30 = 0;
  uint64_t chance_25 = 0;
  uint64_t chance_20 = 0;
  uint64_t chance_15 = 0;
  uint64_t chance_10 = 0;
  uint64_t chance_5 = 0;
  uint64_t chance_0 = 0;
  MatIterator_<unsigned char> pixel_iter, end;
  for (pixel_iter = highest_probability.begin<unsigned char>(),
      end = highest_probability.end<unsigned char>();
       pixel_iter != end; ++pixel_iter) {
    if (pxl < *pixel_iter) {
      pxl = *pixel_iter;
    }
    if (*pixel_iter >= 255) {
      chance_100++;
    }
    if (*pixel_iter > MIN_CHANCE_95) {
      chance_95++;
    }
    if (*pixel_iter > MIN_CHANCE_90) {
      chance_90++;
    }
    if (*pixel_iter > MIN_CHANCE_85) {
      chance_85++;
    }
    if (*pixel_iter > MIN_CHANCE_80) {
      chance_80++;
    }
    if (*pixel_iter > MIN_CHANCE_75) {
      chance_75++;
    }
    if (*pixel_iter > MIN_CHANCE_70) {
      chance_70++;
    }
    if (*pixel_iter > MIN_CHANCE_65) {
      chance_65++;
    }
    if (*pixel_iter > MIN_CHANCE_60) {
      chance_60++;
    }
    if (*pixel_iter > MIN_CHANCE_55) {
      chance_55++;
    }
    if (*pixel_iter > MIN_CHANCE_50) {
      chance_50++;
    }
    if (*pixel_iter > MIN_CHANCE_45) {
      chance_45++;
    }
    if (*pixel_iter > MIN_CHANCE_40) {
      chance_40++;
    }
    if (*pixel_iter > MIN_CHANCE_35) {
      chance_35++;
    }
    if (*pixel_iter > MIN_CHANCE_30) {
      chance_30++;
    }
    if (*pixel_iter > MIN_CHANCE_25) {
      chance_25++;
    }
    if (*pixel_iter > MIN_CHANCE_20) {
      chance_20++;
    }
    if (*pixel_iter > MIN_CHANCE_15) {
      chance_15++;
    }
    if (*pixel_iter > MIN_CHANCE_10) {
      chance_10++;
    }
    if (*pixel_iter > MIN_CHANCE_5) {
      chance_5++;
    }
    if (*pixel_iter > MIN_CHANCE_0) {
      chance_0++;
    }
  }
  cout << "Largest pixel value: " << (uint32_t)pxl;
  cout << "Pixel count: "
       << highest_probability.size().width * highest_probability.size().height
       << endl;
  cout << "Chance " << 255 << ": " << chance_100 << endl;
  cout << "Chance " << (uint32_t)(MIN_CHANCE_95) << ": " << chance_95 << endl;
  cout << "Chance 95: " << chance_90 << endl;
  cout << "Chance 90: " << chance_85 << endl;
  cout << "Chance 85: " << chance_80 << endl;
  cout << "Chance 80: " << chance_75 << endl;
  cout << "Chance 75: " << chance_70 << endl;
  cout << "Chance 70: " << chance_65 << endl;
  cout << "Chance 65: " << chance_60 << endl;
  cout << "Chance 60: " << chance_55 << endl;
  cout << "Chance 55: " << chance_50 << endl;
  cout << "Chance 50: " << chance_45 << endl;
  cout << "Chance 45: " << chance_40 << endl;
  cout << "Chance 40: " << chance_35 << endl;
  cout << "Chance 35: " << chance_30 << endl;
  cout << "Chance 30: " << chance_25 << endl;
  cout << "Chance 25: " << chance_20 << endl;
  cout << "Chance 20: " << chance_15 << endl;
  cout << "Chance 15: " << chance_10 << endl;
  cout << "Chance 10: " << chance_5 << endl;
  cout << "Chance 0: " << chance_0 << endl;

  namedWindow("Probability_dist", WINDOW_NORMAL);
  imshow("Probability_dist", highest_probability);
  waitKey(100);
  namedWindow("Output", WINDOW_NORMAL);
  Mat output_mat(highest_probability.size(), CV_8UC1);
  threshold(highest_probability, output_mat, (double)MIN_CHANCE_25,
            (double)chance_100, CV_THRESH_BINARY);
  imshow("Output", output_mat);
  waitKey(0);
  return output_mat;
}

/*
void GestureRecognition::SkinExtraction(cv::cuda::GpuMat *image,
                                        cv::cuda::GpuMat *skin_filter) {
  cv::Mat image_cpu(*image);
  cv::Mat ycrcb, hsv;
  std::chrono::time_point<std::chrono::system_clock> before_ycrcb =

      std::chrono::system_clock::now();
  cvtColor(image_cpu, ycrcb, CV_BGR2YCrCb);
  std::chrono::time_point<std::chrono::system_clock> after_ycrcb =
      std::chrono::system_clock::now();
  cvtColor(image_cpu, hsv, CV_BGR2HSV);
  std::chrono::time_point<std::chrono::system_clock> after_hsv =
      std::chrono::system_clock::now();
  cv::Mat skin_mask(image_cpu.size(), CV_8UC1);
  MatIterator_<Vec3b> colour_iter, hsv_iter, ycrcb_iter, end;
  MatIterator_<unsigned char> skin_iter;
  for (colour_iter = image_cpu.begin<Vec3b>(), hsv_iter = hsv.begin<Vec3b>(),
      ycrcb_iter = ycrcb.begin<Vec3b>(),
      skin_iter = skin_mask.begin<unsigned char>(),
      end = image_cpu.end<Vec3b>();
       colour_iter != end;
       ++colour_iter, ++hsv_iter, ++ycrcb_iter, ++skin_iter) {
    if ((colour_iter + 1) == (*image).end<Vec3b>())
      cout << "Colours next is end" << endl;
    if ((hsv_iter + 1) == hsv.end<Vec3b>())
      cout << "hsv's next is end" << endl;
    if ((ycrcb_iter + 1) == ycrcb.end<Vec3b>())
      cout << "ycrcbs next is end" << endl;
    if ((skin_iter + 1) == skin_mask.end<unsigned char>())
      cout << "skins next is end" << endl;
Vec3b hsv_pixel = *hsv_iter;
Vec3b ycrcb_pixel = *ycrcb_iter;
Vec3b colour_pixel = *colour_iter;
if (
    // B > 20 and R > B and
    (colour_pixel[0] > 20) && (colour_pixel[2] > colour_pixel[0]) &&
    // R > G and G > 40 and
    (colour_pixel[2] > colour_pixel[1]) && (colour_pixel[1] > 40) &&
    // R > 95 and | R - G | >  15
    (colour_pixel[2] > 95) && ((colour_pixel[2] - colour_pixel[1]) > 15)) {
  // 0.0 <= H <= 50.0 and 0.23 <= S <= 0.68
  if (((hsv_pixel[0] <= 25) && (hsv_pixel[1] >= 58) && (hsv_pixel[1] <= 174))
      // OR
      ||
      // Cr > 135 and Cb > 85 and Y > 80 and
      ((ycrcb_pixel[1] > 135) && (ycrcb_pixel[2] > 85) &&
       (ycrcb_pixel[0] > 80) &&
       // Cr <= (1.5862*Cb) + 20 and
       (ycrcb_pixel[1] <=
        (unsigned char)(1.5862 * (float)ycrcb_pixel[2]) + 20) &&
       // Cr >= (0.3448*Cb) + 76.2069 and
       (ycrcb_pixel[1] <=
        (unsigned char)(0.3448 * (float)ycrcb_pixel[2]) + 76.2069) &&
       // Cr >= (-4.5652*Cb) + 234.5652 and
       (ycrcb_pixel[1] <=
        (unsigned char)(-4.5652 * (float)ycrcb_pixel[2]) + 234.5652) &&
       // Cr <= (-1.15*Cb) + 301.75 and
       (ycrcb_pixel[1] <=
        (unsigned char)(-1.15 * (float)ycrcb_pixel[2]) + 301.75) &&
       // Cr <= (-2.2857*Cb) + 432.85
       (ycrcb_pixel[1] <=
        (unsigned char)(-2.2857 * (float)ycrcb_pixel[2]) + 432.85))) {
    *skin_iter = 255;
  } else {
    *skin_iter = 0;
  }
} else {
  *skin_iter = 0;
}
}
std::chrono::time_point<std::chrono::system_clock> after_loop =
    std::chrono::system_clock::now();

std::chrono::duration<float> ycrcb_t = after_ycrcb - before_ycrcb;
std::chrono::duration<float> hsv_t = after_hsv - after_ycrcb;
std::chrono::duration<float> loop_t = after_loop - after_hsv;

const int ycrcb_ms = ycrcb_t.count() * 1000;
const int hsv_ms = hsv_t.count() * 1000;
const int loop_ms = loop_t.count() * 1000;

cout << "YCrCb: " << ycrcb_ms << endl;
cout << "HSV: " << hsv_ms << endl;
cout << "Skin detect loop: " << loop_ms << endl;
// imshow("Skin mask new", skin_mask);
}
*/

void GestureRecognition::InitializeBackgroundQueue(size_t image_size,
                                                   size_t queue_size) {
  // m_background_image_queue.Initialize(image_size, queue_size);
}

// Entry point
int main() {
  // std::cout << "Hello world." << std::endl;

  VideoCapture cap(0); // open the default camera
  if (!cap.isOpened()) {
    IMAGE_GRAB_EXCEPTION("Failed to open camera.");
  }
  uint64_t count = 0;
  Mat image;

  for (uint32_t i = 0; i < 15; i++) {
    if (!cap.read(image)) {
      IMAGE_GRAB_EXCEPTION("Failed to grab first image.");
    }
  }

  cuda::GpuMat gpu_image;

  GestureRecognition gesture_recognition(image.size().area());
  // esture_recognition.GenerateTrackerbars();
  while (1) {
    try {
      // cout << "Count : " << count++ << endl;

      // read img to pass to class
      Mat image_jpg;
      image_jpg = imread("C:\\Users\\lyndo\\Downloads\\Training_set_"
                         "pictures\\green_wall_both_hands_extended.jpg",
                         CV_LOAD_IMAGE_COLOR);

      if (!cap.read(image)) {
        IMAGE_GRAB_EXCEPTION("Failed to grab image.");
      }

      // cout << "Image (" << image.size << ") " << image.type() << endl;
      // imshow("with face", image);
      gpu_image.upload(image_jpg);
      gesture_recognition.DetectPalm(NULL, &gpu_image);

      // read extra image to flush buffer
      if (!cap.read(image)) {
        IMAGE_GRAB_EXCEPTION("Failed to grab image.");
      }
    } catch (cv::Exception &e) {
      cout << "OpenCV exception: " << e.what();
      break;
    } catch (MemoryException &e) {
      cout << "Memory exception: " << e.what();
      break;
    } catch (ImageGrabException &e) {
      cout << "Image grab exception: " << e.what();
      break;
    } catch (FaceSubtractionException &e) {
      cout << "Face subtraction exception: " << e.what();
      break;
    }
  }

  cout << "Exception caused ending of program." << endl;
  return 0;
}
