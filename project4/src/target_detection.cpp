/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-target_detection.cpp
 * Implements checkerboard corner detection and refinement on camera frames.
 * The output frame is annotated for immediate visual feedback.
 */

#include "target_detection.h"

/**
 * @brief Detect checkerboard corners and draw them on an output frame.
 */
bool detectChessboardCorners(const cv::Mat &frame, const cv::Size &patternSize,
                             std::vector<cv::Point2f> &corners,
                             cv::Mat &outputFrame) {
    if (frame.empty()) {
        return false;
    }

    frame.copyTo(outputFrame);

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    bool found = cv::findChessboardCorners(gray, patternSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH +
                                               cv::CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
        cv::cornerSubPix(
            gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                             30, 0.1));

        cv::drawChessboardCorners(outputFrame, patternSize, corners, found);
    }

    return found;
}