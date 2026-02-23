#include "p3_segmentation.hpp"
#include <cmath>
#include <algorithm>

namespace p3 {

static cv::Mat toGrayscale(const cv::Mat& bgr) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    cv::Mat gray(bgr.rows, bgr.cols, CV_8UC1);

    // Manual BGR -> Gray using ITU-R BT.601 approximation
    // gray = 0.114*B + 0.587*G + 0.299*R
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* src = bgr.ptr<cv::Vec3b>(y);
        uchar* dst = gray.ptr<uchar>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const uchar B = src[x][0];
            const uchar G = src[x][1];
            const uchar R = src[x][2];
            int v = (114 * B + 587 * G + 299 * R) / 1000;
            dst[x] = static_cast<uchar>(std::clamp(v, 0, 255));
        }
    }
    return gray;
}

// Clamp helper for border handling
static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

static cv::Mat gaussianBlur5x5(const cv::Mat& gray) {
    CV_Assert(!gray.empty() && gray.type() == CV_8UC1);

    // A classic 5x5 Gaussian kernel (sigma ~ 1), normalized by 273
    //  1  4  7  4  1
    //  4 16 26 16  4
    //  7 26 41 26  7
    //  4 16 26 16  4
    //  1  4  7  4  1
    static const int K[5][5] = {
        { 1,  4,  7,  4, 1},
        { 4, 16, 26, 16, 4},
        { 7, 26, 41, 26, 7},
        { 4, 16, 26, 16, 4},
        { 1,  4,  7,  4, 1}
    };
    static const int denom = 273;

    cv::Mat out(gray.rows, gray.cols, CV_8UC1);

    for (int y = 0; y < gray.rows; ++y) {
        uchar* dst = out.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                int yy = clampi(y + ky, 0, gray.rows - 1);
                const uchar* row = gray.ptr<uchar>(yy);
                for (int kx = -2; kx <= 2; ++kx) {
                    int xx = clampi(x + kx, 0, gray.cols - 1);
                    sum += K[ky + 2][kx + 2] * row[xx];
                }
            }
            int v = sum / denom;
            dst[x] = static_cast<uchar>(std::clamp(v, 0, 255));
        }
    }
    return out;
}

static double isodataThreshold(const cv::Mat& graySmooth) {
    CV_Assert(graySmooth.type() == CV_8UC1);

    // Start from mid-gray
    double T = 128.0;
    double prevT = -1.0;

    // Iterate until convergence
    while (std::abs(T - prevT) > 0.5) {
        prevT = T;

        double sum1 = 0.0, sum2 = 0.0;
        int cnt1 = 0, cnt2 = 0;

        for (int y = 0; y < graySmooth.rows; ++y) {
            const uchar* row = graySmooth.ptr<uchar>(y);
            for (int x = 0; x < graySmooth.cols; ++x) {
                double v = static_cast<double>(row[x]);
                if (v < T) { sum1 += v; cnt1++; }
                else       { sum2 += v; cnt2++; }
            }
        }

        double m1 = (cnt1 > 0) ? (sum1 / cnt1) : 0.0;
        double m2 = (cnt2 > 0) ? (sum2 / cnt2) : 0.0;

        // New threshold is midpoint between class means
        T = 0.5 * (m1 + m2);
    }

    return T;
}

static cv::Mat applyThreshold(const cv::Mat& graySmooth, double T, bool invert) {
    CV_Assert(graySmooth.type() == CV_8UC1);

    cv::Mat bin(graySmooth.rows, graySmooth.cols, CV_8UC1);

    for (int y = 0; y < graySmooth.rows; ++y) {
        const uchar* src = graySmooth.ptr<uchar>(y);
        uchar* dst = bin.ptr<uchar>(y);
        for (int x = 0; x < graySmooth.cols; ++x) {
            bool isForeground = (src[x] < T); // object is darker than background
            if (invert) {
                // Foreground -> 255 (white), Background -> 0 (black)
                dst[x] = isForeground ? 255 : 0;
            } else {
                // Foreground -> 0 (black), Background -> 255 (white)
                dst[x] = isForeground ? 0 : 255;
            }
        }
    }

    return bin;
}

cv::Mat thresholdBinaryScratch(const cv::Mat& bgr, bool invert) {
    // 1) Grayscale (manual)
    cv::Mat gray = toGrayscale(bgr);

    // 2) Smooth noise (manual Gaussian)
    cv::Mat smooth = gaussianBlur5x5(gray);

    // 3) Adaptive threshold (ISODATA)
    double T = isodataThreshold(smooth);

    // 4) Binary mask (manual)
    return applyThreshold(smooth, T, invert);
}

} // namespace p3