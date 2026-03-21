/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-feature_main.cpp
 * Task 7 entry point for Harris corner detection.
 * This file captures live camera frames, runs Harris response computation,
 * and visualizes both a heat-map overlay and detected corners.
 *
 * Controls:
 *   Threshold trackbar (0-255) : minimum response strength to mark a corner
 *   Block size trackbar (0-4)  : neighbourhood size (maps to 1,3,5,7,9)
 *   +  /  =                    : increase Harris k by 0.005
 *   -                          : decrease Harris k by 0.005
 *   s                          : save screenshot
 *   q                          : quit
 */

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

/**
 * @brief Run Task 7 Harris corner detection with interactive controls.
 * @return 0 on normal exit, -1 if camera cannot be opened.
 */
int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }

    int threshSlider = 150;
    int blockSlider = 1; // maps to blockSize = blockSlider*2+1
    double k = 0.04;

    const std::string win = "Project 4 - Task 7: Harris Corners";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold", win, &threshSlider, 255);
    cv::createTrackbar("Block size idx", win, &blockSlider, 4);

    std::cout << "Harris Corner Detector\n"
              << "Trackbars: Threshold (0-255), Block size index (0-4)\n"
              << "Keys: q=quit  s=save  +/-=k\n";

    cv::Mat frame, small, gray, grayF, response, normResp, display, heat;
    int shotCount = 0;

    // PERFORMANCE: detect on this scale of the original frame
    const double SCALE = 0.4;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // ── Downscale for Harris (fast) ───────────────────────────────────
        cv::resize(frame, small, cv::Size(), SCALE, SCALE);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(grayF, CV_32F);

        int blockSize = blockSlider * 2 + 1;
        cv::cornerHarris(grayF, response, blockSize, 3, k);
        cv::normalize(response, normResp, 0, 255, cv::NORM_MINMAX, CV_32F);

        // ── Build display on full-res frame ───────────────────────────────
        frame.copyTo(display);

        // Heat map overlay (upscale back to full res)
        cv::Mat normU8, heatSmall;
        normResp.convertTo(normU8, CV_8U);
        cv::applyColorMap(normU8, heatSmall, cv::COLORMAP_JET);
        cv::Mat heatFull;
        cv::resize(heatSmall, heatFull, frame.size());
        cv::addWeighted(display, 0.75, heatFull, 0.25, 0, display);

        // Draw circles at corner locations (scale coords back to full res)
        float thr = (float)threshSlider;
        float invS = (float)(1.0 / SCALE);
        int count = 0;
        for (int r = 0; r < normResp.rows; ++r) {
            for (int c = 0; c < normResp.cols; ++c) {
                if (normResp.at<float>(r, c) > thr) {
                    cv::circle(display,
                               cv::Point((int)(c * invS), (int)(r * invS)), 4,
                               cv::Scalar(0, 0, 255), 2);
                    ++count;
                }
            }
        }

        // ── HUD ───────────────────────────────────────────────────────────
        std::ostringstream ks;
        ks << std::fixed << std::setprecision(3) << k;
        cv::putText(display, "Corners: " + std::to_string(count),
                    cv::Point(20, 34), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 0), 2);
        cv::putText(display,
                    "k=" + ks.str() + "  block=" + std::to_string(blockSize) +
                        "  thresh=" + std::to_string(threshSlider),
                    cv::Point(20, 68), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 0), 1);
        cv::putText(display, "q=quit  s=save  +/-=k", cv::Point(20, 100),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200),
                    1);

        cv::imshow(win, display);

        char key = (char)cv::waitKey(10);
        if (key == 'q')
            break;
        if (key == 's') {
            std::filesystem::create_directories("../data/screenshots");
            std::string fname = "../data/screenshots/harris_" +
                                std::to_string(++shotCount) + ".png";
            cv::imwrite(fname, display);
            std::cout << "Saved: " << fname << "\n";
        }
        if (key == '+' || key == '=')
            k = std::min(k + 0.005, 0.30);
        if (key == '-')
            k = std::max(k - 0.005, 0.001);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
