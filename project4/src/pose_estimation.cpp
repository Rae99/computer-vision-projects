#include "pose_estimation.h"
#include <iostream>

bool readCalibrationFromFile(const std::string& filename,
                             cv::Mat& cameraMatrix,
                             cv::Mat& distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        std::cout << "Could not open calibration file: " << filename << std::endl;
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        std::cout << "Calibration data is missing or invalid." << std::endl;
        return false;
    }

    std::cout << "Loaded camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Loaded distortion coefficients:\n" << distCoeffs << std::endl;

    return true;
}

bool estimateBoardPose(const std::vector<cv::Vec3f>& worldPoints,
                       const std::vector<cv::Point2f>& imagePoints,
                       const cv::Mat& cameraMatrix,
                       const cv::Mat& distCoeffs,
                       cv::Mat& rvec,
                       cv::Mat& tvec) {
    if (worldPoints.empty() || imagePoints.empty()) {
        return false;
    }

    if (worldPoints.size() != imagePoints.size()) {
        std::cout << "worldPoints size does not match imagePoints size." << std::endl;
        return false;
    }

    bool success = cv::solvePnP(
        worldPoints,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec
    );

    return success;
}