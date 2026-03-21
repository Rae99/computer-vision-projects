/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-camera_calibration.cpp
 * Implements camera calibration and file output functions.
 * It computes intrinsic parameters from checkerboard correspondences
 * and stores calibration results for later pose estimation.
 */

#include "camera_calibration.h"
#include <filesystem>
#include <iostream>

/**
 * @brief Calibrate camera intrinsics and distortion from sample sets.
 */
bool runCameraCalibration(
    const std::vector<std::vector<cv::Vec3f>> &pointList,
    const std::vector<std::vector<cv::Point2f>> &cornerList,
    const cv::Size &imageSize, cv::Mat &cameraMatrix, cv::Mat &distCoeffs,
    std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs,
    double &reprojectionError) {
    if (pointList.size() < 5 || cornerList.size() < 5) {
        std::cout << "Need at least 5 calibration samples." << std::endl;
        return false;
    }

    if (pointList.size() != cornerList.size()) {
        std::cout << "pointList size does not match cornerList size."
                  << std::endl;
        return false;
    }

    cameraMatrix = (cv::Mat_<double>(3, 3) << 1.0, 0.0,
                    static_cast<double>(imageSize.width) / 2.0, 0.0, 1.0,
                    static_cast<double>(imageSize.height) / 2.0, 0.0, 0.0, 1.0);

    distCoeffs = cv::Mat();
    rvecs.clear();
    tvecs.clear();

    std::cout << "\nCamera matrix before calibration:\n"
              << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients before calibration:\n"
              << distCoeffs << std::endl;

    int flags = cv::CALIB_FIX_ASPECT_RATIO;

    reprojectionError =
        cv::calibrateCamera(pointList, cornerList, imageSize, cameraMatrix,
                            distCoeffs, rvecs, tvecs, flags);

    std::cout << "\nCamera matrix after calibration:\n"
              << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients after calibration:\n"
              << distCoeffs << std::endl;
    std::cout << "Final reprojection error: " << reprojectionError << std::endl;

    std::cout << "Number of rotation vectors: " << rvecs.size() << std::endl;
    std::cout << "Number of translation vectors: " << tvecs.size() << std::endl;

    return true;
}

/**
 * @brief Write camera calibration values to a YAML file.
 */
bool writeCalibrationToFile(const std::string &filename,
                            const cv::Mat &cameraMatrix,
                            const cv::Mat &distCoeffs,
                            double reprojectionError) {
    std::filesystem::path outputPath(filename);
    std::filesystem::create_directories(outputPath.parent_path());

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cout << "Could not open file for writing: " << filename
                  << std::endl;
        return false;
    }

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << reprojectionError;
    fs.release();

    std::cout << "Calibration written to file: " << filename << std::endl;
    return true;
}