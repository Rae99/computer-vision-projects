// p3_segmentation.cpp
#include "p3_segmentation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace p3 {

static inline unsigned char clampU8(int v) {
    return static_cast<unsigned char>(std::max(0, std::min(255, v)));
}

/**
 * Convert BGR (CV_8UC3) to grayscale (CV_8UC1) from scratch.
 * Uses ITU-R BT.601 luma approximation:
 *   gray = 0.114 * B + 0.587 * G + 0.299 * R
 */
static cv::Mat bgrToGrayScratch(const cv::Mat &bgr) {
    if (bgr.empty())
        return cv::Mat();

    if (bgr.type() == CV_8UC1) {
        // Already grayscale
        return bgr.clone();
    }

    if (bgr.type() != CV_8UC3) {
        throw std::runtime_error(
            "bgrToGrayScratch expects CV_8UC3 or CV_8UC1 input.");
    }

    cv::Mat gray(bgr.rows, bgr.cols, CV_8UC1);

    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b *src = bgr.ptr<cv::Vec3b>(y);
        unsigned char *dst = gray.ptr<unsigned char>(y);

        for (int x = 0; x < bgr.cols; ++x) {
            const int B = src[x][0];
            const int G = src[x][1];
            const int R = src[x][2];

            // Integer-friendly rounding:
            // 0.114=114/1000, 0.587=587/1000, 0.299=299/1000
            // Add 500 for rounding.
            const int v = (114 * B + 587 * G + 299 * R + 500) / 1000;
            dst[x] = clampU8(v);
        }
    }

    return gray;
}

/**
 * 5-tap Gaussian kernel: [1 4 6 4 1] / 16
 * Implemented as separable filter (horizontal then vertical) from scratch.
 * Border handling: replicate.
 */
static cv::Mat gaussianBlur5x5Scratch(const cv::Mat &gray) {
    if (gray.empty())
        return cv::Mat();
    if (gray.type() != CV_8UC1) {
        throw std::runtime_error(
            "gaussianBlur5x5Scratch expects CV_8UC1 input.");
    }

    const int k[5] = {1, 4, 6, 4, 1};
    const int denom = 16;

    // Horizontal pass into 16-bit to reduce rounding loss
    cv::Mat tmp(gray.rows, gray.cols, CV_16SC1);

    for (int y = 0; y < gray.rows; ++y) {
        const unsigned char *src = gray.ptr<unsigned char>(y);
        short *dst = tmp.ptr<short>(y);

        for (int x = 0; x < gray.cols; ++x) {
            int sum = 0;
            for (int i = -2; i <= 2; ++i) {
                int xx = x + i;
                if (xx < 0)
                    xx = 0;
                if (xx >= gray.cols)
                    xx = gray.cols - 1;
                sum += k[i + 2] * static_cast<int>(src[xx]);
            }
            // keep as int-ish in tmp; defer /16 to later (or do here)
            dst[x] = static_cast<short>(sum); // still scaled by 1 (not divided)
        }
    }

    // Vertical pass back to 8-bit, apply both denom factors (16*16 = 256)
    cv::Mat out(gray.rows, gray.cols, CV_8UC1);

    for (int y = 0; y < gray.rows; ++y) {
        unsigned char *dst = out.ptr<unsigned char>(y);

        for (int x = 0; x < gray.cols; ++x) {
            int sum = 0;
            for (int i = -2; i <= 2; ++i) {
                int yy = y + i;
                if (yy < 0)
                    yy = 0;
                if (yy >= gray.rows)
                    yy = gray.rows - 1;

                const short *row = tmp.ptr<short>(yy);
                sum += k[i + 2] * static_cast<int>(row[x]);
            }

            // tmp stored horizontal sum without /16.
            // Now vertical sum multiplies by another kernel and we divide by
            // 16*16 = 256.
            const int v = (sum + 128) / 256; // +128 for rounding
            dst[x] = clampU8(v);
        }
    }

    return out;
}

/**
 * ISODATA (iterative intermeans) thresholding from scratch.
 * Returns a threshold T in [0,255].
 */
static double isodataThresholdScratch(const cv::Mat &gray) {
    if (gray.empty())
        return 128.0;
    if (gray.type() != CV_8UC1) {
        throw std::runtime_error(
            "isodataThresholdScratch expects CV_8UC1 input.");
    }

    double T = 128.0;
    double prevT = -1.0;

    // Reasonable guard to prevent infinite loops on weird inputs.
    const int kMaxIters = 100;

    for (int iter = 0; iter < kMaxIters; ++iter) {
        if (std::abs(T - prevT) <= 0.5)
            break;
        prevT = T;

        double sum1 = 0.0, sum2 = 0.0;
        int cnt1 = 0, cnt2 = 0;

        for (int y = 0; y < gray.rows; ++y) {
            const unsigned char *row = gray.ptr<unsigned char>(y);
            for (int x = 0; x < gray.cols; ++x) {
                const int v = row[x];
                if (v < T) {
                    sum1 += v;
                    ++cnt1;
                } else {
                    sum2 += v;
                    ++cnt2;
                }
            }
        }

        // If one side is empty, fallback to previous threshold.
        if (cnt1 == 0 || cnt2 == 0)
            break;

        const double m1 = sum1 / cnt1;
        const double m2 = sum2 / cnt2;
        T = 0.5 * (m1 + m2);
    }

    // Clamp
    if (T < 0.0)
        T = 0.0;
    if (T > 255.0)
        T = 255.0;
    return T;
}

cv::Mat thresholdBinary(const cv::Mat &bgr) {
    if (bgr.empty())
        return cv::Mat();

    // 1) Grayscale (from scratch)
    cv::Mat gray = bgrToGrayScratch(bgr);

    // 2) Smooth (from scratch)
    cv::Mat smooth = gaussianBlur5x5Scratch(gray);

    // 3) Auto threshold (from scratch)
    const double T = isodataThresholdScratch(smooth);

    // 4) Binary segmentation (from scratch)
    // Convention (WRITE-ONCE / no flags):
    //   foreground/object = 255 (non-zero)
    //   background        = 0
    //
    // Because object is darker than background:
    //   intensity < T  => object => 255
    //   else           => background => 0
    cv::Mat binary(smooth.rows, smooth.cols, CV_8UC1);

    for (int y = 0; y < smooth.rows; ++y) {
        const unsigned char *src = smooth.ptr<unsigned char>(y);
        unsigned char *dst = binary.ptr<unsigned char>(y);

        for (int x = 0; x < smooth.cols; ++x) {
            dst[x] = (src[x] < T) ? 255 : 0;
        }
    }

    return binary;
}

} // namespace p3