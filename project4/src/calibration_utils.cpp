/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-calibration_utils.cpp
 * Implements small utility functions for checkerboard geometry and
 * calibration image persistence.
 * These helpers are shared across calibration and pose pipelines.
 */

#include "calibration_utils.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

/**
 * @brief Build checkerboard world points on the Z=0 plane.
 */
std::vector<cv::Vec3f> buildChessboardWorldPoints(const cv::Size &patternSize) {
    std::vector<cv::Vec3f> pointSet;

    for (int row = 0; row < patternSize.height; row++) {
        for (int col = 0; col < patternSize.width; col++) {
            pointSet.emplace_back(static_cast<float>(col),
                                  static_cast<float>(-row), 0.0f);
        }
    }

    return pointSet;
}

/**
 * @brief Save a calibration frame to disk using a numbered filename.
 */
bool saveCalibrationImage(const cv::Mat &image, const std::string &folderPath,
                          int imageIndex) {
    if (image.empty()) {
        std::cout << "saveCalibrationImage: image is empty." << std::endl;
        return false;
    }

    std::filesystem::create_directories(folderPath);

    std::ostringstream filename;
    filename << folderPath << "/calib_" << std::setw(2) << std::setfill('0')
             << imageIndex << ".png";

    std::string fullPath = filename.str();
    std::cout << "Trying to save image to: " << fullPath << std::endl;

    bool success = cv::imwrite(fullPath, image);

    if (!success) {
        std::cout << "cv::imwrite failed." << std::endl;
    }

    return success;
}