/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-calibration_utils.h
 * Declares utility helpers used by calibration and pose modules.
 * These helpers generate checkerboard world coordinates and save
 * calibration frames to disk.
 */

#ifndef CALIBRATION_UTILS_H
#define CALIBRATION_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Build 3D world points for a checkerboard target.
 * @param patternSize Number of inner corners as (columns, rows).
 * @return Ordered checkerboard world points with z = 0.
 */
std::vector<cv::Vec3f> buildChessboardWorldPoints(const cv::Size &patternSize);

/**
 * @brief Save a calibration image with a deterministic filename.
 * @param image Input image to save.
 * @param folderPath Destination folder path.
 * @param imageIndex Sequential index appended to the filename.
 * @return True if the image is saved successfully.
 */
bool saveCalibrationImage(const cv::Mat &image, const std::string &folderPath,
                          int imageIndex);

#endif