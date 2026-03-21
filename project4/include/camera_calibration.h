/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-camera_calibration.h
 * Declares camera calibration routines for estimating intrinsics and
 * distortion from checkerboard correspondences.
 * Also provides helpers for writing calibration results to file.
 */

#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Run camera calibration from matched 3D-2D checkerboard points.
 * @param pointList 3D world points for each calibration image.
 * @param cornerList 2D detected corners for each calibration image.
 * @param imageSize Calibration image size.
 * @param cameraMatrix Output intrinsic matrix.
 * @param distCoeffs Output distortion coefficients.
 * @param rvecs Output rotation vectors for each view.
 * @param tvecs Output translation vectors for each view.
 * @param reprojectionError Output average reprojection error.
 * @return True if calibration ran successfully.
 */
bool runCameraCalibration(
    const std::vector<std::vector<cv::Vec3f>> &pointList,
    const std::vector<std::vector<cv::Point2f>> &cornerList,
    const cv::Size &imageSize, cv::Mat &cameraMatrix, cv::Mat &distCoeffs,
    std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs,
    double &reprojectionError);

/**
 * @brief Write calibration results to a YAML file.
 * @param filename Output file path.
 * @param cameraMatrix Intrinsic matrix to write.
 * @param distCoeffs Distortion coefficients to write.
 * @param reprojectionError Final reprojection error value.
 * @return True if the file is written successfully.
 */
bool writeCalibrationToFile(const std::string &filename,
                            const cv::Mat &cameraMatrix,
                            const cv::Mat &distCoeffs,
                            double reprojectionError);

#endif