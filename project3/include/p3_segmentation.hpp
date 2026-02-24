#ifndef P3_SEGMENTATION_HPP
#define P3_SEGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

namespace p3 {

    // ---------- Task1 ----------
    cv::Mat thresholdBinary(const cv::Mat &bgr);

    // ---------- Task2 ----------
    cv::Mat morphCleanup(const cv::Mat &bin, int openKernel = 3, int closeKernel = 11);

    // Optional helper (you already had)
    cv::Mat keepLargestComponent(const cv::Mat &binary);

    // ---------- Task3 ----------
    std::vector<int> selectTopRegionsByArea(const cv::Mat &stats, int nLabels,
                                            int minArea, int maxRegions);

    cv::Mat regionMapColor(const cv::Mat &cleanBinary,
                           int minArea = 500,
                           int connectivity = 8,
                           int maxRegions = 10);

    // A more stable selection strategy for Task4 (recommended)
    std::vector<int> selectRegionsByScore(const cv::Mat &stats,
                                          const cv::Mat &centroids,
                                          int imgW, int imgH,
                                          int minArea,
                                          int maxRegions,
                                          double centerWeight = 0.002,
                                          bool rejectBorderTouching = true);

    // ---------- Task4 ----------
    struct RegionFeatures {
        int regionId = -1;
        int area = 0;
        cv::Point2d centroid{0.0, 0.0};

        double thetaMajor = 0.0;
        double thetaMinor = 0.0;

        double obbWidth = 0.0;
        double obbHeight = 0.0;
        std::array<cv::Point2f, 4> obbCorners{};

        cv::Point2f axisP1{};
        cv::Point2f axisP2{};

        double percentFilled = 0.0;
        double aspectRatio = 0.0;

        std::vector<double> featureVector() const {
            return {percentFilled, aspectRatio};
        }
    };

    RegionFeatures computeRegionFeatures(const cv::Mat &labels, int regionId);
    void drawRegionOverlay(cv::Mat &canvas, const RegionFeatures &f);

} // namespace p3

#endif