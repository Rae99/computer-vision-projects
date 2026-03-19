#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

bool readCalibrationFromFile(const std::string& filename,
                             cv::Mat& cameraMatrix,
                             cv::Mat& distCoeffs);

bool estimateBoardPose(const std::vector<cv::Vec3f>& worldPoints,
                       const std::vector<cv::Point2f>& imagePoints,
                       const cv::Mat& cameraMatrix,
                       const cv::Mat& distCoeffs,
                       cv::Mat& rvec,
                       cv::Mat& tvec);

#endif