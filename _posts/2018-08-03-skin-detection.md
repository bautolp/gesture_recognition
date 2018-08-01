---
layout: post
title: Skin Detection
subtitle: Using face detection to profile and detect skin
#bigimg: /img/path.jpg
---

## Steps
The steps to perform the skin detection are:
1. Detect the face
2. Profile the face
3. Find probability that other pixels are skin based on face
4. Threshold probability to obtain skin mask

Note, before the initial image enters the skin detection a bilateral filter is applied on it to reduce noise but preserve edges

## Face Detection Using Classifier
- Face detection is done using a HAAR cascade classifier
- This returns the size and location of the face

## Profiling Skin
- The face is profiled in the YCrCb colour space
- To profile the face, the face detection area is shrunk and a smaller rectangle including only skin is obtained
- The rectangle is fed into a histogram
- The initial image is compared to the histogram to generate a back projection

## Converting Back Projection to a Binary Image
- The back projection is compared to a threshold and a binary skin mask is generated
- Erosing/dilation are often applied in this step, but do not seem to be necessary with this type of skin detection

## Limitations
- Bright spectral lighting may cause failures
- If hands intersect face, they will mesh with face contour