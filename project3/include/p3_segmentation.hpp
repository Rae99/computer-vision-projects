#ifndef P3_SEGMENTATION_HPP
#define P3_SEGMENTATION_HPP

#include <opencv2/opencv.hpp>

namespace p3 {

/**
 * @brief Segment a dark object from a light background.
 *
 * The function produces a binary mask where:
 *      - Foreground (object)  = 255 (non-zero)
 *      - Background           = 0
 *
 * Assumptions (as required by the project):
 *      - Object is darker than the background
 *      - Scene is reasonably well-lit
 *      - Only intensity is used for separation
 *
 * This function performs:
 *      1. Grayscale conversion
 *      2. Gaussian smoothing (implemented manually in .cpp)
 *      3. Automatic threshold selection using ISODATA
 *      4. Binary segmentation (foreground = non-zero)
 *
 * @param bgr  Input color image (BGR)
 * @return     CV_8UC1 binary mask (foreground = 255)
 */
cv::Mat thresholdBinary(const cv::Mat &bgr);

} // namespace p3

#endif