---
layout: post
title: Gesture Recognition
subtitle: Using convexity analysis to determine gestures
#bigimg: /img/path.jpg
---

## Algorithm Overview
Using the width, height, and location of the face, the contours of the skin mask are analysed.

## Contour filter
Contours are filtered and any contours which are much lower than the face, very small in area compared to the face, or intersect with the face, are discarded.

## Convexity Analysis
Convexity analysis is performed by obtaining the convex hull and convexity defects of each contour. If you are unfamiliar with these terms, they are worth reading up on.
The following points on the contour are determined (note relative measurements are calculated using the size of the face):
1. Highest
2. Leftmost within a finger tips vertical distance of the highest
3. Rightmost within a finger tips vertical distance of the highest
4. Leftmost above the wrist
5. Rightmost above the wrist

The contour is then re-examined using this information to attempt to locate the thumb. This is done by checking the leftmost and rightmost points above the wrist. If they are at least half a thumbs length farther out than the leftmost and rightmost points within a finger tips vertical distance of the highest, they are marked as a potential thumb.
Potential thumbs are re-examined, checking the nearby points, checking how the contour varies vertically and horizontally. If the contour travels far horizontally before having a vertical wall, it is marked as a thumb. 

## Defect Filtering
The defects which fall into the following categories are discarded:
1. Far point is above the start and end points
2. Any point is much lower than the highest point
3. Distance from start/end point to far point is too small
4. Distance from start/end point to far point is too large

## Defect Analysis
After the presence of the thumb is determined, the defects of the contour are analysed.
The following handlers for defects exist:
1. Thumb
- If the defect set is the thumb, it is skipped
2. Two fingers
- If the start and end points of a defect set are within a fingertips vertical distance of each other, they are labelled as two finger tips
3. Pinky finger
- If the two fingers handler does not catch the defect set as two fingers, the pinky finger handler is executed
- The leftmost and rightmost point in the defect set are compared to the leftmost and rightmost point above the wrist
    - If either of them are the same point, then the defect is labelled as a pinky
- If the pinky is found as one of the points, the other points vertical position is checked
    - If the other point is above the pinky, it is also labelled as a finger tip
4. Single finger
- If the other handlers fail the single finger handler is executed
- The single finger handler just labels the highest point of the defect set as a finger

## Limitations
- The skin mask must work, if it failed, this analysis will fail
- Sideways gestures are unreliable
- If hand is at an angle with the camera, the reliability drops (contours start to mesh together)
- Hand cannot intersect face
- Hand must be near head level

## Future Work
- Work on the detection parameters to make sure they are properly tuned
    - Many parameters were guessed at, worked, and never revised
        - Could be some bugs hiding deep in the logic
- Adding angle related checks on the defects could provide more reliability
- Additional constraints to classify the hand
