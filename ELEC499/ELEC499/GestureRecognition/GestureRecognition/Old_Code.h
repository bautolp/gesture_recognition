#pragma once

///		BACKGROUND SUBTRACTION

// Mat foreground_mask; // fg mask generated by MOG2 method
// Ptr<BackgroundSubtractor> mog2_background_subtractor;
// mog2_background_subtractor =
// createBackgroundSubtractorMOG2(BACKGROUND_QUEUE_SIZE, 48.0, false);
// Mat image;
// for (size_t image_count = 0; image_count < 0; image_count++) {
//	if (!cap.read(image)) {
//		IMAGE_GRAB_EXCEPTION("Failed to grab image.");
//	}
// namedWindow("Initial Image", CV_WINDOW_AUTOSIZE);
// imshow("Initial Image", image);
// waitKey(1);

// update the background model
//	mog2_background_subtractor->apply(image, foreground_mask, -1);
//}
// Mat background;
// mog2_background_subtractor->getBackgroundImage(background);

// namedWindow("background", CV_WINDOW_AUTOSIZE);
// imshow("background", background);
// mog2_background_subtractor->apply(image, foreground_mask, -1);
// cout << "Foreground mask (" << foreground_mask.size << ") "
//     << foreground_mask.type() << endl;

// namedWindow("foreground", CV_WINDOW_AUTOSIZE);
// imshow("foreground", foreground_mask);
// waitKey(0);

// Mat masked_img;
// bitwise_and(image, cv::Scalar(255, 255, 255), masked_img,
//            foreground_mask);
// namedWindow("After mask", CV_WINDOW_AUTOSIZE);
// imshow("After mask", masked_img);
// waitKey(0);

///			DRAW CONTOURS
// Mat drawing = Mat::zeros((*image).size(), CV_8UC3);
// for (int i = 0; i < contours->size(); i++) {
//  int colour = 100;
// drawContours(drawing, *contours, i, 100, 2, 8, hierarchy, 0, Point());
//}
#ifdef _DEBUG
// namedWindow("Contours", CV_WINDOW_AUTOSIZE);
// imshow("Contours", drawing);
// waitKey(0);
#endif

///			LOG CHROMATICITY

// void GestureRecognition::LogChromaticity2D(cv::Mat *image,
//	cv::Mat *chomaticity) {

/*  Mat image(600, 600, CV_8UC3, Scalar(127, 127, 127));

int cn = (*image).channels();
uint8_t *pixelPtr = (uint8_t *)(*image).data;

for (int i = 0; i < (*image).rows; i++) {
for (int j = 0; j < (*image).cols; j++) {
Scalar_<uint8_t> bgrPixel;
bgrPixel.val[0] = pixelPtr[i * (*image).cols * cn + j * cn + 0]; // B
bgrPixel.val[1] = pixelPtr[i * (*image).cols * cn + j * cn + 1]; // G
bgrPixel.val[2] = pixelPtr[i * (*image).cols * cn + j * cn + 2]; // R
if (bgrPixel.val[2] != 0) { // avoid division by zero
float a = (*chomaticity).cols / 2 +
50 * (log((float)bgrPixel.val[0] / (float)bgrPixel.val[2]));
float b = (*chomaticity).rows / 2 +
50 * (log((float)bgrPixel.val[1] / (float)bgrPixel.val[2]));
if (!isinf(a) && !isinf(b))
(*chomaticity).at<Vec3b>(a, b) = Vec3b(255, 2, 3);
}
}
}

imshow("log-chroma", image);
imwrite("log-chroma.png", image);
waitKey(0);*/
//}

///			CASCADE DISPLAY
/*


void GestureRecognition::DisplayCascade(string cascade_name, Mat *image) {
CascadeClassifier cascade;
String cascade_name_full =
string("C:"
"\\Users\\lyndo\\Developer\\ELEC499\\GestureRecogn"
"ition\\bin\\") +
cascade_name;
if (!cascade.load(cascade_name_full)) {
FACE_SUBTRACTION_EXCEPTION("Failed to load %s.", cascade_name_full.c_str());
}

Mat gray, img_copy;
cvtColor(*image, gray, CV_BGR2GRAY);
equalizeHist(gray, gray);

img_copy = (*image).clone();
std::vector<Rect> cascade_rectangles;
cascade.detectMultiScale(gray, cascade_rectangles, 1.1, 2,
0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

// Draw circles on the detected faces
for (int i = 0; i < cascade_rectangles.size(); i++) {
Point center(
int(cascade_rectangles[i].x + cascade_rectangles[i].width * 0.5),
int(cascade_rectangles[i].y + cascade_rectangles[i].height * 0.5));
ellipse(img_copy, center,
Size(int(cascade_rectangles[i].width * 0.5),
int(cascade_rectangles[i].height * 0.5)),
0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

namedWindow(cascade_name.c_str(), 1);
imshow(cascade_name.c_str(), img_copy);
waitKey(-1);
}
}
*/

/// OTHER CODE PORTED IN FOR PALM DETECTION
/*
void GestureRecognition::FindHand(
        cv::Mat *image) { // Find the contours in the foreground

        vector<pair<Point, double>> palm_centers;
        vector<vector<Point>> contours;
        findContours(*image, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        for (int i = 0; i < contours.size(); i++) {
                // Ignore all small insignificant areas
                if (contourArea(contours[i]) < 5000) {
                        continue;
                }
                // Draw contour
                vector<vector<Point>> tcontours;
                tcontours.push_back(contours[i]);
                drawContours(*image, tcontours, -1, cv::Scalar(0, 0, 255), 2);

                // Detect Hull in current contour
                vector<vector<Point>> hulls(1);
                vector<vector<int>> hullsI(1);
                convexHull(Mat(tcontours[0]), hulls[0], false);
                convexHull(Mat(tcontours[0]), hullsI[0], false);
                drawContours(*image, hulls, -1, cv::Scalar(0, 255, 0), 2);

                // Find minimum area rectangle to enclose hand
                RotatedRect rect = minAreaRect(Mat(tcontours[0]));

                // Find Convex Defects
                vector<Vec4i> defects;
                if (hullsI[0].size() <= 0) {
                        continue;
                }
                Point2f rect_points[4];
                rect.points(rect_points);
                for (int j = 0; j < 4; j++) {
                        cout << "Drawing line" << endl;
                        cv::line(*image, rect_points[j], rect_points[(j + 1) %
4], Scalar(255, 0, 0), 1, 8);
                }
                Point rough_palm_center;
                convexityDefects(tcontours[0], hullsI[0], defects);
                if (defects.size() < 3) {
                        std::cout << "< 3 defects" << endl;
                        continue;
                }
                vector<Point> palm_points;
                for (int j = 0; j < defects.size(); j++) {
                        int startidx = defects[j][0];
                        Point ptStart(tcontours[0][startidx]);
                        int endidx = defects[j][1];
                        Point ptEnd(tcontours[0][endidx]);
                        int faridx = defects[j][2];
                        Point ptFar(tcontours[0][faridx]);
                        // Sum up all the hull and defect points to compute
average rough_palm_center += ptFar + ptStart + ptEnd;
                        palm_points.push_back(ptFar);
                        palm_points.push_back(ptStart);
                        palm_points.push_back(ptEnd);
                }

                // Get palm center by 1st getting the average of all defect
points,
                // this is the rough palm center, Then U chose the closest 3
points
                // ang get the circle radius and center formed from them which
is the
                // palm center.
                rough_palm_center.x /= defects.size() * 3;
                rough_palm_center.y /= defects.size() * 3;
                Point closest_pt = palm_points[0];
                vector<pair<double, int>> distvec;
                for (int i = 0; i < palm_points.size(); i++) {
                        cout << "Getting distance" << endl;
                        distvec.push_back(make_pair(dist(rough_palm_center,
palm_points[i]), i));
                }
                sort(distvec.begin(), distvec.end());

                // Keep choosing 3 points till you find a circle with a valid
radius
                // As there is a high chance that the closes points might be in
a
                // linear line or too close that it forms a very large circle
                pair<Point, double> soln_circle;
                for (int i = 0; i + 2 < distvec.size(); i++) {
                        Point p1 = palm_points[distvec[i + 0].second];
                        Point p2 = palm_points[distvec[i + 1].second];
                        Point p3 = palm_points[distvec[i + 2].second];
                        cout << "Getting circle from points line" << endl;
                        soln_circle = circleFromPoints(p1, p2, p3); // Final
palm center,radius if (soln_circle.second != 0) break;
                }

                // Find avg palm centers for the last few frames to stabilize
its
                // centers, also find the avg radius
                palm_centers.push_back(soln_circle);
                if (palm_centers.size() > 10) {
                        cout << "erasing from palm center" << endl;
                        palm_centers.erase(palm_centers.begin());
                }
                Point palm_center;
                double radius = 0;
                for (int i = 0; i < palm_centers.size(); i++) {
                        palm_center += palm_centers[i].first;
                        radius += palm_centers[i].second;
                }
                palm_center.x /= palm_centers.size();
                palm_center.y /= palm_centers.size();
                radius /= palm_centers.size();

                // Draw the palm center and the palm circle
                // The size of the palm gives the depth of the hand
                cv::circle(*image, palm_center, 5, Scalar(144, 144, 255), 3);
                cv::circle(*image, palm_center, radius, Scalar(144, 144, 255),
2);

                // Detect fingers by finding points that form an almost
isosceles
                // triangle with certain thesholds
                int no_of_fingers = 0;
                for (int j = 0; j < defects.size(); j++) {
                        int startidx = defects[j][0];
                        Point ptStart(tcontours[0][startidx]);
                        int endidx = defects[j][1];
                        Point ptEnd(tcontours[0][endidx]);
                        int faridx = defects[j][2];
                        Point ptFar(tcontours[0][faridx]);
                        // X o--------------------------o Y
                        double Xdist = sqrt(dist(palm_center, ptFar));
                        double Ydist = sqrt(dist(palm_center, ptStart));
                        double length = sqrt(dist(ptFar, ptStart));

                        double retLength = sqrt(dist(ptEnd, ptFar));
                        // Play with these thresholds to improve performance
                        if (length <= 3 * radius && Ydist >= 0.4 * radius &&
length >= 10 && retLength >= 10 && max(length, retLength) / min(length,
retLength) >= 0.8) if (min(Xdist, Ydist) / max(Xdist, Ydist) <= 0.8) { if
((Xdist >= 0.1 * radius && Xdist <= 1.3 * radius && Xdist < Ydist) || (Ydist >=
0.1 * radius && Ydist <= 1.3 * radius && Xdist > Ydist)) { cout << "Drawing line
at end" << endl; line(*image, ptEnd, ptFar, Scalar(0, 255, 0), 1),
no_of_fingers++;
                                        }
                                }
                }

                no_of_fingers = min(5, no_of_fingers);
                cout << "Number OF FINGERS: " << no_of_fingers << endl;
        }

        cout << "Displaying image." << endl;
        namedWindow("Hand detect output", 1);
        imshow("Hand detect output", *image);
}



// This function returns the radius and the center of the circle given 3 points
// If a circle cannot be formed , it returns a zero radius circle centered at
// (0,0)
pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3) {
        double offset = pow(p2.x, 2) + pow(p2.y, 2);
        double bc = (pow(p1.x, 2) + pow(p1.y, 2) - offset) / 2.0;
        double cd = (offset - pow(p3.x, 2) - pow(p3.y, 2)) / 2.0;
        double det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y -
p2.y); double TOL = 0.0000001; if (abs(det) < TOL) { cout << "POINTS TOO CLOSE"
<< endl; return make_pair(Point(0, 0), 0);
        }

        double idet = 1 / det;
        double centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
        double centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
        double radius = sqrt(pow(p2.x - centerx, 2) + pow(p2.y - centery, 2));

        return make_pair(Point(centerx, centery), radius);
}

*/
