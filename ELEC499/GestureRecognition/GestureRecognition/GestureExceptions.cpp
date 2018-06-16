#include "GestureExceptions.h"
#include <sstream>

MemoryException::~MemoryException() throw() {}

const char *MemoryException::what() const throw() { return msg.c_str(); }

void MemoryException::AppendFileAndLine(const char *file, int line) {
  std::ostringstream full_msg;
  full_msg << msg << " (Line " << line << ": " << file << ")";
  msg = full_msg.str();
}

ImageGrabException::~ImageGrabException() throw() {}

const char *ImageGrabException::what() const throw() { return msg.c_str(); }

void ImageGrabException::AppendFileAndLine(const char *file, int line) {
  std::ostringstream full_msg;
  full_msg << msg << " (Line " << line << ": " << file << ")";
  msg = full_msg.str();
}

FaceSubtractionException::~FaceSubtractionException() throw() {}

const char *FaceSubtractionException::what() const throw() {
  return msg.c_str();
}

void FaceSubtractionException::AppendFileAndLine(const char *file, int line) {
  std::ostringstream full_msg;
  full_msg << msg << " (Line " << line << ": " << file << ")";
  msg = full_msg.str();
}
