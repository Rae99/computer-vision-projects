#ifndef CALIBRATION_UTILS_H
#define CALIBRATION_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::vector<cv::Vec3f> buildChessboardWorldPoints(const cv::Size& patternSize);

bool saveCalibrationImage(const cv::Mat& image,
                          const std::string& folderPath,
                          int imageIndex);

#endif