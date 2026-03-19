#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

bool runCameraCalibration(
    const std::vector<std::vector<cv::Vec3f>>& pointList,
    const std::vector<std::vector<cv::Point2f>>& cornerList,
    const cv::Size& imageSize,
    cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs,
    std::vector<cv::Mat>& rvecs,
    std::vector<cv::Mat>& tvecs,
    double& reprojectionError
);

bool writeCalibrationToFile(const std::string& filename,
                            const cv::Mat& cameraMatrix,
                            const cv::Mat& distCoeffs,
                            double reprojectionError);

#endif