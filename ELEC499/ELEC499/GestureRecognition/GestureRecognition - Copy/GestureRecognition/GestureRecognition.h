#ifndef GestureRecognition_H
#define GestureRecognition_H
#include "BackgroundImageQueue.h"
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <string>

class GestureRecognition {
public:
  GestureRecognition(uint64_t image_payload_size);
  ~GestureRecognition();

  bool DetectPalm(cv::Point *location, cv::cuda::GpuMat *image);
  void PushBackgroundImage(cv::cuda::GpuMat *image);
  void GenerateTrackerbars();

private:
  void BackgroundSubtraction(cv::cuda::GpuMat *image);
  bool ContourExtraction(cv::cuda::GpuMat *image);
  void BlurImage(cv::cuda::GpuMat *image);
  void ThresholdImage(cv::cuda::GpuMat *image);
  void FaceSubtraction(cv::cuda::GpuMat *image);
  cv::Mat SkinExtraction(cv::cuda::GpuMat *image);
  void DisplayCascade(std::string cascade_name, cv::cuda::GpuMat *image);
  void LogChromaticity2D(cv::cuda::GpuMat *image,
                         cv::cuda::GpuMat *chomaticity);
  void FindHand(cv::cuda::GpuMat *image);
  size_t FindContours(cv::cuda::GpuMat *image,
                      std::vector<std::vector<cv::Point>> *contours);
  void InitializeBackgroundQueue(size_t image_size, size_t queue_size);
  BackgroundImageQueue m_background_image_queue;

  struct RunTimeOptions {
    uint64_t background_image_count;
    uint64_t image_payload_size;
    bool dynamic_background_subtraction;
  };
  RunTimeOptions m_runtime_options;
  cv::Ptr<cv::cuda::CascadeClassifier> m_face_cascade;
  cv::Ptr<cv::cuda::CascadeClassifier> m_eye_cascade;
};

#endif
