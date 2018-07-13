#include "BackgroundImageQueue.h"
#include "GestureExceptions.h"
#include <iostream>

BackgroundImageQueue::BackgroundImageQueue()
    : m_max_image_count(0), m_image_size(0), m_size(0), m_head(NULL),
      m_tail(NULL) {}

BackgroundImageQueue::~BackgroundImageQueue() {
  uint64_t n = 0;
  for (BackgroundImageNode *node = m_head; node != m_tail; node = node->next) {
    std::cout << "Deleting " << n++ << std::endl;
    if (node->prev == NULL) {
      std::cout << "Previous node is NULL" << std::endl;
    }
    if (node->next == NULL) {
      std::cout << "Next node is NULL" << std::endl;
    }
    delete node;
  }
  std::cout << "Deleting tail" << std::endl;
  if (m_tail != NULL) {
    delete m_tail;
  }
}

void BackgroundImageQueue::Push(void *image) {
  // Make sure there is an actual image payload
  if (image == NULL) {
    MEMORY_EXCEPTION("Image passed to push function is NULL.");
  }
  if ((m_head == NULL) || (m_tail == NULL)) {
    MEMORY_EXCEPTION("Must allocate memory for queue before pushing.");
  }

  memcpy(m_tail->image, image, m_image_size);

  // Because queue is circular, head->prev is tail, and tail->prev is
  m_tail = m_tail->prev;
  m_head = m_head->prev;

  if (m_size < m_max_image_count)
    m_size++;
}

void BackgroundImageQueue::Initialize(uint64_t image_size,
                                      uint64_t image_count) {
  m_max_image_count = image_count;
  m_image_size = image_size;
  BackgroundImageNode *prev = NULL;
  BackgroundImageNode *next = NULL;
  for (uint64_t image_idx = 0; image_idx < m_max_image_count; image_idx++) {
    BackgroundImageNode *temp = new BackgroundImageNode;
    temp->image = (void *)new char[m_image_size];

    // prev->prev = prev is valid for any node but the head, so overwrite this
    // in the tail assignment block, because the head->prev is the tail
    temp->prev = prev;
    if (image_idx == 0) {
      m_head = temp;
    } else if (image_idx == (m_max_image_count - 1)) {
      m_tail = temp;
      m_tail->next = m_head;
      m_head->prev = m_tail;
      prev->next = m_tail;
    } else {
      prev->next = temp;
    }
    if (prev != NULL) {
      prev->next = temp;
    }
    prev = temp;
  }
}

bool BackgroundImageQueue::IsFull() { return (m_max_image_count == m_size); }
