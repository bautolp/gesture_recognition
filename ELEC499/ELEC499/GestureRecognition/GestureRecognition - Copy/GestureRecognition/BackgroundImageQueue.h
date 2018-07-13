#ifndef BackgroundImageQueue_H
#define BackgroundImageQueue_H

#include <stdint.h>

class BackgroundImageQueue {
public:
  BackgroundImageQueue::BackgroundImageQueue();
  BackgroundImageQueue::~BackgroundImageQueue();

  void BackgroundImageQueue::Push(void *);
  bool BackgroundImageQueue::IsFull();
  void Initialize(uint64_t image_size, uint64_t image_count);

private:
  struct BackgroundImageNode {
    void *image;
    BackgroundImageNode *next;
    BackgroundImageNode *prev;
    BackgroundImageNode::BackgroundImageNode() {
      image = NULL;
      next = NULL;
      prev = NULL;
    }
    BackgroundImageNode::~BackgroundImageNode() {
      if (image != NULL) {
        delete[] image;
        image = NULL;
      }
    }
  };

  uint64_t m_size;
  uint64_t m_max_image_count;
  uint64_t m_image_size;
  BackgroundImageNode *m_head;
  BackgroundImageNode *m_tail;
};

#endif
