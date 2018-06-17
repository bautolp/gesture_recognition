#ifndef PalmTracker_H
#define PalmTracker_H

#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>

#include <stdint.h>

#define MAX_PALMS_SIZE 100

class PalmTracker {
public:
  PalmTracker::PalmTracker();
  PalmTracker::~PalmTracker();

  void PalmTracker::SetCurrentLocation(cv::Point current_location);
  cv::Point PalmTracker::GetCurrentLocation();

private:
  cv::Point m_palm_locations[MAX_PALMS_SIZE];
  uint32_t m_write_location;
};

#endif
