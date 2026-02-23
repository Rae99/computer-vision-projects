#pragma once
#include <opencv2/opencv.hpp>

namespace p3 {

/**
 * Task 1:
 * Segment foreground object from background using adaptive thresholding.
 *
 * Input:  BGR image from camera
 * Output: Binary image (object = white, background = black)
 */
cv::Mat thresholdBinary(const cv::Mat& bgr);

}