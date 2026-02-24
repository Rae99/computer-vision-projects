#include "p3_segmentation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace p3 {

// ---------------------
// Task1 helpers
// ---------------------
static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

static cv::Mat toGrayscale(const cv::Mat &bgr) {
    CV_Assert(!bgr.empty() && bgr.type() == CV_8UC3);

    cv::Mat gray(bgr.rows, bgr.cols, CV_8UC1);

    // Manual BGR -> Gray (BT.601 approx)
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

    // 5x5 Gaussian kernel (sigma ~ 1), normalized by 273
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

        // midpoint between class means
        T = 0.5 * (m1 + m2);
    }

    return T;
}

static cv::Mat applyThreshold(const cv::Mat &graySmooth, double T) {
    CV_Assert(graySmooth.type() == CV_8UC1);

    cv::Mat bin(graySmooth.rows, graySmooth.cols, CV_8UC1);

    for (int y = 0; y < graySmooth.rows; ++y) {
        const uchar *src = graySmooth.ptr<uchar>(y);
        uchar *dst = bin.ptr<uchar>(y);
        for (int x = 0; x < graySmooth.cols; ++x) {
            bool isForeground = (src[x] < T); // dark object on light background
            dst[x] = isForeground ? 255 : 0;  // foreground=255
        }
    }
    return bin;
}

// ---------------------
// Task1
// ---------------------
cv::Mat thresholdBinary(const cv::Mat &bgr) {
    cv::Mat gray = toGrayscale(bgr);
    cv::Mat smooth = gaussianBlur5x5(gray);
    double T = isodataThreshold(smooth);
    return applyThreshold(smooth, T);
}

// ---------------------
// Task2 (OpenCV morphology)
// ---------------------
cv::Mat morphCleanup(const cv::Mat &bin, int openKernel, int closeKernel) {
    CV_Assert(!bin.empty() && bin.type() == CV_8UC1);

    cv::Mat k1 = cv::getStructuringElement(cv::MORPH_RECT,
                                           cv::Size(openKernel, openKernel));
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_RECT,
                                           cv::Size(closeKernel, closeKernel));

    cv::Mat opened, closed;
    cv::morphologyEx(bin, opened, cv::MORPH_OPEN, k1);
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, k2);
    return closed;
}

// Optional helper
cv::Mat keepLargestComponent(const cv::Mat &binary) {
    CV_Assert(!binary.empty() && binary.type() == CV_8UC1);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(binary, labels, stats, centroids,
                                             8, CV_32S);
    if (n <= 1) {
        return cv::Mat::zeros(binary.size(), CV_8UC1);
    }

    int bestLabel = 1;
    int bestArea = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) {
            bestArea = area;
            bestLabel = i;
        }
    }

    cv::Mat out = cv::Mat::zeros(binary.size(), CV_8UC1);
    out.setTo(255, labels == bestLabel);
    return out;
}

// ---------------------
// Task3 helpers
// ---------------------
std::vector<int> selectTopRegionsByArea(const cv::Mat &stats, int nLabels,
                                        int minArea, int maxRegions) {
    std::vector<std::pair<int, int>> items; // (area, label)
    items.reserve(std::max(0, nLabels - 1));

    for (int i = 1; i < nLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= minArea) {
            items.push_back({area, i});
        }
    }

    std::sort(items.begin(), items.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<int> out;
    for (int i = 0; i < (int)items.size(); ++i) {
        if (maxRegions > 0 && (int)out.size() >= maxRegions) break;
        out.push_back(items[i].second);
    }
    return out;
}

// More stable selection strategy for Task4
std::vector<int> selectRegionsByScore(const cv::Mat &stats,
                                      const cv::Mat &centroids,
                                      int imgW, int imgH,
                                      int minArea,
                                      int maxRegions,
                                      double centerWeight,
                                      bool rejectBorderTouching) {
    std::vector<std::pair<double, int>> items; // (score, label)

    double cx = (imgW - 1) * 0.5;
    double cy = (imgH - 1) * 0.5;

    int nLabels = stats.rows;

    for (int i = 1; i < nLabels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < minArea) continue;

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        if (rejectBorderTouching) {
            bool touches = (x == 0) || (y == 0) || (x + w >= imgW) || (y + h >= imgH);
            if (touches) continue;
        }

        double mx = centroids.at<double>(i, 0);
        double my = centroids.at<double>(i, 1);
        double dist2 = (mx - cx) * (mx - cx) + (my - cy) * (my - cy);

        // Score: prefer large area, penalize far from center
        double score = (double)area - centerWeight * dist2;
        items.push_back({score, i});
    }

    std::sort(items.begin(), items.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<int> out;
    for (int k = 0; k < (int)items.size(); ++k) {
        if (maxRegions > 0 && (int)out.size() >= maxRegions) break;
        out.push_back(items[k].second);
    }
    return out;
}

cv::Mat regionMapColor(const cv::Mat &cleanBinary,
                       int minArea,
                       int connectivity,
                       int maxRegions) {
    CV_Assert(!cleanBinary.empty() && cleanBinary.type() == CV_8UC1);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(cleanBinary, labels, stats,
                                             centroids, connectivity, CV_32S);

    cv::Mat colored(cleanBinary.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    if (n <= 1) {
        return colored;
    }

    std::vector<int> keep = selectTopRegionsByArea(stats, n, minArea, maxRegions);

    std::vector<char> isKept(n, 0);
    for (int id : keep) {
        if (id >= 1 && id < n) isKept[id] = 1;
    }

    // deterministic palette (no flicker across runs)
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(50, 255);
    std::vector<cv::Vec3b> palette(n, cv::Vec3b(0, 0, 0));
    for (int i = 1; i < n; ++i) {
        palette[i] = cv::Vec3b((uchar)dist(rng), (uchar)dist(rng), (uchar)dist(rng));
    }

    for (int y = 0; y < labels.rows; ++y) {
        const int *row = labels.ptr<int>(y);
        cv::Vec3b *dst = colored.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) {
            int id = row[x];
            if (id > 0 && id < n && isKept[id]) {
                dst[x] = palette[id];
            } else {
                dst[x] = cv::Vec3b(0, 0, 0);
            }
        }
    }

    return colored;
}

// ---------------------
// Task4 (HAND-WRITTEN feature extraction)
// ---------------------
static inline void rotatePoint(double x, double y, double c, double s,
                               double &xr, double &yr) {
    xr = c * x - s * y;
    yr = s * x + c * y;
}

RegionFeatures computeRegionFeatures(const cv::Mat &labels, int regionId) {
    CV_Assert(!labels.empty() && labels.type() == CV_32S);

    RegionFeatures f;
    f.regionId = regionId;

    // 1) area + centroid
    double m00 = 0.0, m10 = 0.0, m01 = 0.0;

    for (int y = 0; y < labels.rows; ++y) {
        const int *row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == regionId) {
                m00 += 1.0;
                m10 += x;
                m01 += y;
            }
        }
    }

    f.area = (int)m00;
    if (f.area <= 0) return f;

    f.centroid.x = m10 / m00;
    f.centroid.y = m01 / m00;

    // 2) second central moments (region-based)
    double mu20 = 0.0, mu02 = 0.0, mu11 = 0.0;

    for (int y = 0; y < labels.rows; ++y) {
        const int *row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == regionId) {
                double dx = x - f.centroid.x;
                double dy = y - f.centroid.y;
                mu20 += dx * dx;
                mu02 += dy * dy;
                mu11 += dx * dy;
            }
        }
    }

    // covariance-like normalization
    double a = mu20 / m00;
    double c = mu02 / m00;
    double b = mu11 / m00;

    // 3) principal axis
    double theta = 0.5 * std::atan2(2.0 * b, (a - c));
    f.thetaMajor = theta;
    f.thetaMinor = f.thetaMajor + CV_PI / 2.0;

    // 4) oriented bounding box by rotating pixels into major-axis frame
    double ct = std::cos(f.thetaMajor);
    double st = std::sin(f.thetaMajor);

    double minX = 1e18, maxX = -1e18, minY = 1e18, maxY = -1e18;

    for (int y = 0; y < labels.rows; ++y) {
        const int *row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == regionId) {
                double dx = x - f.centroid.x;
                double dy = y - f.centroid.y;

                // rotate into major-axis frame: R(-theta)
                double xr, yr;
                rotatePoint(dx, dy, ct, -st, xr, yr);

                minX = std::min(minX, xr);
                maxX = std::max(maxX, xr);
                minY = std::min(minY, yr);
                maxY = std::max(maxY, yr);
            }
        }
    }

    f.obbWidth = (maxX - minX);
    f.obbHeight = (maxY - minY);

    std::array<cv::Point2f, 4> cornersR = {
        cv::Point2f((float)minX, (float)minY),
        cv::Point2f((float)maxX, (float)minY),
        cv::Point2f((float)maxX, (float)maxY),
        cv::Point2f((float)minX, (float)maxY)
    };

    // rotate back: R(+theta) + centroid
    for (int i = 0; i < 4; ++i) {
        double xr = cornersR[i].x;
        double yr = cornersR[i].y;

        double xBack, yBack;
        rotatePoint(xr, yr, ct, st, xBack, yBack);
        f.obbCorners[i] = cv::Point2f((float)(xBack + f.centroid.x),
                                      (float)(yBack + f.centroid.y));
    }

    // 5) features
    double obbArea = std::max(1e-9, f.obbWidth * f.obbHeight);
    f.percentFilled = (double)f.area / obbArea;

    double w = std::max(1e-9, f.obbWidth);
    double h = std::max(1e-9, f.obbHeight);
    double ratio = h / w;
    f.aspectRatio = (ratio >= 1.0) ? ratio : (1.0 / ratio);

    // 6) least central moment axis segment for drawing (minor axis)
    double len = 0.5 * std::max(f.obbWidth, f.obbHeight);
    double cm = std::cos(f.thetaMinor);
    double sm = std::sin(f.thetaMinor);

    f.axisP1 = cv::Point2f((float)(f.centroid.x - len * cm),
                           (float)(f.centroid.y - len * sm));
    f.axisP2 = cv::Point2f((float)(f.centroid.x + len * cm),
                           (float)(f.centroid.y + len * sm));

    return f;
}

void drawRegionOverlay(cv::Mat &canvas, const RegionFeatures &f) {
    if (canvas.empty() || f.area <= 0) return;

    // OBB
    for (int i = 0; i < 4; ++i) {
        cv::line(canvas, f.obbCorners[i], f.obbCorners[(i + 1) % 4],
                 cv::Scalar(0, 255, 255), 2);
    }

    // least-moment axis
    cv::line(canvas, f.axisP1, f.axisP2, cv::Scalar(255, 0, 0), 2);

    // show feature values near centroid
    char buf[256];
    std::snprintf(buf, sizeof(buf), "filled=%.3f  ar=%.3f",
                  f.percentFilled, f.aspectRatio);

    cv::Point org((int)std::round(f.centroid.x) + 10,
                  (int)std::round(f.centroid.y) - 10);
    cv::putText(canvas, buf, org, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);
}

} // namespace p3