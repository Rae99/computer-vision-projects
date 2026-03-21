/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-target_detection.h
 * Declares checkerboard corner detection used by calibration and AR tasks.
 * The interface returns refined corner positions and an annotated output
 * frame for visualization.
 */

#ifndef TARGET_DETECTION_H
#define TARGET_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Detect and refine checkerboard corners in one frame.
 * @param frame Input BGR frame.
 * @param patternSize Number of inner corners as (columns, rows).
 * @param corners Output refined corner list.
 * @param outputFrame Output frame with optional corner overlay.
 * @return True if a checkerboard is found.
 */
bool detectChessboardCorners(const cv::Mat &frame, const cv::Size &patternSize,
                             std::vector<cv::Point2f> &corners,
                             cv::Mat &outputFrame);

#endif