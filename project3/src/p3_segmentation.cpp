#include "p3_segmentation.hpp"
#include <algorithm>
#include <cmath>

namespace p3 {

// Clamp helper for border handling
static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

static cv::Mat toGrayscale(const cv::Mat &bgr) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    cv::Mat gray(bgr.rows, bgr.cols, CV_8UC1);

    // Manual BGR -> Gray using ITU-R BT.601 approximation
    // gray = 0.114*B + 0.587*G + 0.299*R
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b *src = bgr.ptr<cv::Vec3b>(y);
        uchar *dst = gray.ptr<uchar>(y);
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

static cv::Mat gaussianBlur5x5(const cv::Mat &gray) {
    CV_Assert(!gray.empty() && gray.type() == CV_8UC1);

    // A classic 5x5 Gaussian kernel (sigma ~ 1), normalized by 273
    //  1  4  7  4  1
    //  4 16 26 16  4
    //  7 26 41 26  7
    //  4 16 26 16  4
    //  1  4  7  4  1
    static const int K[5][5] = {{1, 4, 7, 4, 1},
                                {4, 16, 26, 16, 4},
                                {7, 26, 41, 26, 7},
                                {4, 16, 26, 16, 4},
                                {1, 4, 7, 4, 1}};
    static const int denom = 273;

    cv::Mat out(gray.rows, gray.cols, CV_8UC1);

    for (int y = 0; y < gray.rows; ++y) {
        uchar *dst = out.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            int sum = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                int yy = clampi(y + ky, 0, gray.rows - 1);
                const uchar *row = gray.ptr<uchar>(yy);
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

static double isodataThreshold(const cv::Mat &graySmooth) {
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
            const uchar *row = graySmooth.ptr<uchar>(y);
            for (int x = 0; x < graySmooth.cols; ++x) {
                double v = static_cast<double>(row[x]);
                if (v < T) {
                    sum1 += v;
                    cnt1++;
                } else {
                    sum2 += v;
                    cnt2++;
                }
            }
        }

        double m1 = (cnt1 > 0) ? (sum1 / cnt1) : 0.0;
        double m2 = (cnt2 > 0) ? (sum2 / cnt2) : 0.0;

        // New threshold is midpoint between class means
        T = 0.5 * (m1 + m2);
    }

    return T;
}

static cv::Mat applyThreshold(const cv::Mat &graySmooth, double T,
                              bool invert) {
    CV_Assert(graySmooth.type() == CV_8UC1);

    cv::Mat bin(graySmooth.rows, graySmooth.cols, CV_8UC1);

    for (int y = 0; y < graySmooth.rows; ++y) {
        const uchar *src = graySmooth.ptr<uchar>(y);
        uchar *dst = bin.ptr<uchar>(y);
        for (int x = 0; x < graySmooth.cols; ++x) {
            bool isForeground =
                (src[x] < T); // object is darker than background
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

cv::Mat thresholdBinary(const cv::Mat &bgr) {
    // 1) Grayscale (manual)
    cv::Mat gray = toGrayscale(bgr);

    // 2) Smooth noise (manual Gaussian)
    cv::Mat smooth = gaussianBlur5x5(gray);

    // 3) Adaptive threshold (ISODATA)
    double T = isodataThreshold(smooth);

    // 4) Binary mask (manual)
    return applyThreshold(smooth, T, true);
}

// in p3_segmentation.cpp

// Single erosion pass: a white pixel stays white only if ALL neighbors in
// kernel are white
// static cv::Mat erode(const cv::Mat &bin, int kernelSize) {
//     CV_Assert(bin.type() == CV_8UC1);
//     int r = kernelSize / 2;
//     cv::Mat out(bin.rows, bin.cols, CV_8UC1, cv::Scalar(0));

//     for (int y = 0; y < bin.rows; ++y) {
//         uchar *dst = out.ptr<uchar>(y);
//         for (int x = 0; x < bin.cols; ++x) {
//             // Only care about foreground pixels
//             if (bin.at<uchar>(y, x) == 0)
//                 continue;

//             bool allFg = true;
//             for (int ky = -r; ky <= r && allFg; ++ky) {
//                 int yy = clampi(y + ky, 0, bin.rows - 1);
//                 for (int kx = -r; kx <= r && allFg; ++kx) {
//                     int xx = clampi(x + kx, 0, bin.cols - 1);
//                     if (bin.at<uchar>(yy, xx) == 0)
//                         allFg = false;
//                 }
//             }
//             dst[x] = allFg ? 255 : 0;
//         }
//     }
//     return out;
// }

// // Single dilation pass: a black pixel becomes white if ANY neighbor in
// kernel
// // is white
// static cv::Mat dilate(const cv::Mat &bin, int kernelSize) {
//     CV_Assert(bin.type() == CV_8UC1);
//     int r = kernelSize / 2;
//     cv::Mat out(bin.rows, bin.cols, CV_8UC1, cv::Scalar(0));

//     for (int y = 0; y < bin.rows; ++y) {
//         uchar *dst = out.ptr<uchar>(y);
//         for (int x = 0; x < bin.cols; ++x) {
//             // If already foreground, keep it
//             if (bin.at<uchar>(y, x) == 255) {
//                 dst[x] = 255;
//                 continue;
//             }

//             bool anyFg = false;
//             for (int ky = -r; ky <= r && !anyFg; ++ky) {
//                 int yy = clampi(y + ky, 0, bin.rows - 1);
//                 for (int kx = -r; kx <= r && !anyFg; ++kx) {
//                     int xx = clampi(x + kx, 0, bin.cols - 1);
//                     if (bin.at<uchar>(yy, xx) == 255)
//                         anyFg = true;
//                 }
//             }
//             dst[x] = anyFg ? 255 : 0;
//         }
//     }
//     return out;
// }

// Public function: opening (remove noise) then closing (fill holes)
// from scratch using our manual erosion/dilation
// cv::Mat morphCleanup(const cv::Mat &bin, int openKernel, int closeKernel) {
//     cv::Mat opened = dilate(erode(bin, openKernel), openKernel);
//     cv::Mat closed = erode(dilate(opened, closeKernel), closeKernel);
//     return closed;
// }

cv::Mat morphCleanup(const cv::Mat &bin, int openKernel, int closeKernel) {
    cv::Mat k1 = cv::getStructuringElement(cv::MORPH_RECT,
                                           cv::Size(openKernel, openKernel));
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_RECT,
                                           cv::Size(closeKernel, closeKernel));

    cv::Mat opened, closed;
    cv::morphologyEx(bin, opened, cv::MORPH_OPEN, k1);
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, k2);
    return closed;
}

cv::Mat keepLargestComponent(const cv::Mat &binary) {
    CV_Assert(!binary.empty());
    CV_Assert(binary.type() == CV_8UC1);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(binary, labels, stats, centroids,
                                             8, CV_32S);
    if (n <= 1)
        return cv::Mat::zeros(binary.size(), CV_8UC1);

    const int W = binary.cols;
    const int H = binary.rows;
    const double cx = (W - 1) * 0.5;
    const double cy = (H - 1) * 0.5;

    int bestLabel = -1;
    double bestScore = -1e18;

    // You can tune these two
    const int MIN_AREA = (W * H) / 500; // ignore tiny blobs
    const double CENTER_WEIGHT = 0.002; // penalty for being far from center

    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < MIN_AREA)
            continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Reject any component that touches the image boundary
        bool touches = (x == 0) || (y == 0) || (x + w >= W) || (y + h >= H);
        if (touches)
            continue;

        double mx = centroids.at<double>(i, 0);
        double my = centroids.at<double>(i, 1);
        double dist2 = (mx - cx) * (mx - cx) + (my - cy) * (my - cy);

        // Score: prefer large area, prefer near center
        double score = (double)area - CENTER_WEIGHT * dist2;

        if (score > bestScore) {
            bestScore = score;
            bestLabel = i;
        }
    }

    // Fallback: if everything touched the border, just use largest (rare)
    if (bestLabel < 0) {
        int bestArea = 0;
        for (int i = 1; i < n; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > bestArea) {
                bestArea = area;
                bestLabel = i;
            }
        }
    }

    cv::Mat out = cv::Mat::zeros(binary.size(), CV_8UC1);
    out.setTo(255, labels == bestLabel);
    return out;
}

} // namespace p3