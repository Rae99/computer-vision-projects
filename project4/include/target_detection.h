#ifndef TARGET_DETECTION_H
#define TARGET_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

bool detectChessboardCorners(const cv::Mat& frame,
                             const cv::Size& patternSize,
                             std::vector<cv::Point2f>& corners,
                             cv::Mat& outputFrame);

#endif