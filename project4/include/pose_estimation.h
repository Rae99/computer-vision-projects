/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-pose_estimation.h
 * Declares utilities for loading camera calibration and estimating
 * checkerboard pose relative to the camera.
 * These functions are used by AR rendering tasks in Project 4.
 */

#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Load camera intrinsics and distortion from a calibration file.
 * @param filename Input calibration file path.
 * @param cameraMatrix Output intrinsic matrix.
 * @param distCoeffs Output distortion coefficients.
 * @return True if calibration data is loaded correctly.
 */
bool readCalibrationFromFile(const std::string &filename, cv::Mat &cameraMatrix,
                             cv::Mat &distCoeffs);

/**
 * @brief Estimate checkerboard pose from 3D-2D correspondences.
 * @param worldPoints Checkerboard 3D world points.
 * @param imagePoints Detected 2D image points.
 * @param cameraMatrix Camera intrinsic matrix.
 * @param distCoeffs Distortion coefficients.
 * @param rvec Output rotation vector.
 * @param tvec Output translation vector.
 * @return True if pose estimation succeeds.
 */
bool estimateBoardPose(const std::vector<cv::Vec3f> &worldPoints,
                       const std::vector<cv::Point2f> &imagePoints,
                       const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                       cv::Mat &rvec, cv::Mat &tvec);

#endif