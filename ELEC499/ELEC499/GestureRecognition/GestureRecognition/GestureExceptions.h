#ifndef GestureExceptions_H
#define GestureExceptions_H

#include <memory>
#include <string>

#define MEMORY_EXCEPTION(str, ...)                                             \
  (throw MemoryException(__FILE__, __LINE__, (str), ##__VA_ARGS__))

class MemoryException : public std::exception {
public:
#pragma warning(push)
#pragma warning(disable : 4996)
  template <typename... Args>
  inline MemoryException( char *file, int line,  std::string &message,
                         Args... args) {
    size_t size = _snprintf(nullptr, 0, message.c_str(), args...);
    std::unique_ptr<char[]> buf(new char[size]);
    _snprintf(buf.get(), size, message.c_str(), args...);
    msg = std::string(buf.get(), buf.get() + size - 1);
  }
#pragma warning(pop)
  ~MemoryException() throw();
  virtual  char *what()  throw();

private:
  std::string msg;
  void AppendFileAndLine( char *file, int line);
};

#define IMAGE_GRAB_EXCEPTION(str, ...)                                         \
  (throw ImageGrabException(__FILE__, __LINE__, (str), ##__VA_ARGS__))

class ImageGrabException : public std::exception {
public:
#pragma warning(push)
#pragma warning(disable : 4996)
  template <typename... Args>
  inline ImageGrabException( char *file, int line,
                             std::string &message, Args... args) {
    size_t size = _snprintf(nullptr, 0, message.c_str(), args...);
    std::unique_ptr<char[]> buf(new char[size]);
    _snprintf(buf.get(), size, message.c_str(), args...);
    msg = std::string(buf.get(), buf.get() + size - 1);
  }
#pragma warning(pop)
  ~ImageGrabException() throw();
  virtual  char *what()  throw();

private:
  std::string msg;
  void AppendFileAndLine( char *file, int line);
};

#define FACE_SUBTRACTION_EXCEPTION(str, ...)                                   \
  (throw FaceSubtractionException(__FILE__, __LINE__, (str), ##__VA_ARGS__))

class FaceSubtractionException : public std::exception {
public:
#pragma warning(push)
#pragma warning(disable : 4996)
  template <typename... Args>
  inline FaceSubtractionException( char *file, int line,
                                   std::string &message, Args... args) {
    size_t size = _snprintf(nullptr, 0, message.c_str(), args...);
    std::unique_ptr<char[]> buf(new char[size]);
    _snprintf(buf.get(), size, message.c_str(), args...);
    msg = std::string(buf.get(), buf.get() + size - 1);
  }
#pragma warning(pop)
  ~FaceSubtractionException() throw();
  virtual  char *what()  throw();

private:
  std::string msg;
  void AppendFileAndLine( char *file, int line);
};

#endif
