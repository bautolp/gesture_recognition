#include "PalmTracker.h"

PalmTracker::PalmTracker() { m_write_location = 0; }
PalmTracker::~PalmTracker() {}

void PalmTracker::SetCurrentLocation(cv::Point current_location) {
  // Should apply some form of filtering here to reduce noise

  m_palm_locations[m_write_location++] =
      current_location; // Make sure this copies and doesn't set pointer
  m_write_location = m_write_location % MAX_PALMS_SIZE;
}

cv::Point PalmTracker::GetCurrentLocation() {
  if (m_write_location == 0)
    return m_palm_locations[MAX_PALMS_SIZE - 1];
  else
    return m_palm_locations[m_write_location - 1];
}
