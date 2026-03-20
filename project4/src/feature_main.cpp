// feature_main.cpp
// Task 7 – Robust Feature Detection using Harris Corner Detector
//
// This program opens the camera and detects Harris corners on every frame.
// Three parameters can be tuned at runtime:
//   • Threshold slider (0-255)  – how strong a response counts as a corner
//   • Block size    slider (1-5) – neighbourhood size (mapped to odd value)
//   • k param  (+/- keys)       – Harris sensitivity (typical range 0.04-0.10)
//
// Controls:
//   q         – quit
//   s         – save screenshot to ../data/screenshots/harris_N.png
//   + / =     – increase k by 0.005
//   -         – decrease k by 0.005
// ===========================================================================

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

int main() {
    std::cout << "Current working directory: "
              << std::filesystem::current_path() << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera." << std::endl;
        return -1;
    }

    // ---- Tunable parameters ----
    // threshSlider: raw trackbar value 0-255.
    //   The actual threshold is applied against the Harris response
    //   normalised to [0, 255].
    int threshSlider = 150;

    // blockSizeSlider: 1-5 (mapped to odd kernel sizes 1,3,5,7,9).
    //   Larger block → smoother response, fewer false positives.
    int blockSizeSlider = 1;   // default maps to blockSize=3

    // k: Harris detector free parameter (typical 0.04 – 0.10).
    //   Lower k → more sensitive (more corners found).
    //   Higher k → less sensitive (only strong corners).
    double k = 0.04;

    // apertureSize for the Sobel gradient inside Harris (must be 3, 5, or 7)
    const int apertureSize = 3;

    // ---- Window + trackbars ----
    const std::string winName = "Project 4 - Task 7: Harris Corners";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold",  winName, &threshSlider, 255);
    cv::createTrackbar("Block size (idx)", winName, &blockSizeSlider, 4);

    std::cout << "Harris Corner Detector\n"
              << "Trackbars: Threshold (0-255), Block size index (0=1,1=3,2=5,3=7,4=9)\n"
              << "Keys: q=quit  s=save  +/-=adjust k\n" << std::endl;

    cv::Mat frame, gray, grayFloat, harrisResponse, harrisNorm, display;
    int screenshotCount = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: empty frame." << std::endl;
            break;
        }

        // Map slider index to an odd block size: 0→1, 1→3, 2→5, 3→7, 4→9
        int blockSize = blockSizeSlider * 2 + 1;

        // ---- Harris corner detection ----
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(grayFloat, CV_32F);

        cv::cornerHarris(grayFloat, harrisResponse,
                         blockSize, apertureSize, k);

        // Normalise response to [0, 255] for thresholding
        cv::normalize(harrisResponse, harrisNorm, 0, 255,
                      cv::NORM_MINMAX, CV_32F);

        // ---- Build display image ----
        frame.copyTo(display);

        // Overlay a semi-transparent heat map of the response strength
        cv::Mat heatMap;
        harrisNorm.convertTo(heatMap, CV_8U);
        cv::applyColorMap(heatMap, heatMap, cv::COLORMAP_JET);
        cv::addWeighted(display, 0.75, heatMap, 0.25, 0, display);

        // Draw a circle at every pixel that exceeds the threshold
        float threshold = static_cast<float>(threshSlider);
        int cornerCount = 0;
        for (int r = 0; r < harrisNorm.rows; r++) {
            for (int c = 0; c < harrisNorm.cols; c++) {
                if (harrisNorm.at<float>(r, c) > threshold) {
                    cv::circle(display, cv::Point(c, r), 4,
                               cv::Scalar(0, 0, 255), 2);
                    cornerCount++;
                }
            }
        }

        // ---- HUD overlay ----
        // Format k with 3 decimal places
        std::ostringstream kStr;
        kStr << std::fixed << std::setprecision(3) << k;

        std::string line1 = "Corners detected: " + std::to_string(cornerCount);
        std::string line2 = "k=" + kStr.str()
                          + "  blockSize=" + std::to_string(blockSize)
                          + "  thresh=" + std::to_string(threshSlider);
        std::string line3 = "q=quit  s=save  +/-=k";

        cv::putText(display, line1, cv::Point(20,  30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, line2, cv::Point(20,  65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);
        cv::putText(display, line3, cv::Point(20,  95),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

        cv::imshow(winName, display);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'q') break;

        if (key == 's') {
            std::filesystem::create_directories("../data/screenshots");
            std::string fname = "../data/screenshots/harris_"
                + std::to_string(++screenshotCount) + ".png";
            cv::imwrite(fname, display);
            std::cout << "Saved screenshot: " << fname << std::endl;
        }

        // Adjust Harris k parameter
        if (key == '+' || key == '=') {
            k = std::min(k + 0.005, 0.30);
            std::cout << "k = " << std::fixed << std::setprecision(3) << k << std::endl;
        }
        if (key == '-') {
            k = std::max(k - 0.005, 0.001);
            std::cout << "k = " << std::fixed << std::setprecision(3) << k << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
