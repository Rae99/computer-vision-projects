// compile: g++ entropy_map.cpp -o entropy_map `pkg-config --cflags --libs
// opencv4`

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

double computeEntropy(const cv::Mat &patch) {
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = range;
    cv::Mat hist;
    cv::calcHist(&patch, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    hist /= (float)patch.total();

    double entropy = 0.0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 0)
            entropy -= p * log2(p);
    }
    return entropy;
}

cv::Mat computeEntropyMap(const cv::Mat &grayImg, int windowSize = 15) {
    int rows = grayImg.rows;
    int cols = grayImg.cols;
    int half = windowSize / 2;

    // Pad image so output has same size as input (reflect padding)
    cv::Mat padded;
    cv::copyMakeBorder(grayImg, padded, half, half, half, half,
                       cv::BORDER_REFLECT);

    cv::Mat entropyMap(rows, cols, CV_32F, cv::Scalar(0));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Extract local window
            cv::Rect roi(c, r, windowSize, windowSize);
            cv::Mat patch = padded(roi);
            entropyMap.at<float>(r, c) = (float)computeEntropy(patch);
        }
        // Progress indicator
        if (r % 50 == 0)
            std::cout << "Processing row " << r << "/" << rows << std::endl;
    }
    return entropyMap;
}

int main() {
    cv::Mat img = cv::imread("../images/photo.jpg");
    if (img.empty()) {
        std::cerr << "Could not load image!" << std::endl;
        return -1;
    }

    // Resize to speed things up
    cv::resize(img, img, cv::Size(400, 300));

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute entropy map (window size = 15x15 pixels)
    cv::Mat entropyMap = computeEntropyMap(gray, 15);

    // Normalize to 0-255 for visualization
    cv::Mat entropyVis;
    cv::normalize(entropyMap, entropyVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply "hot" colormap (dark=low, bright=high entropy)
    cv::Mat entropyColor;
    cv::applyColorMap(entropyVis, entropyColor, cv::COLORMAP_HOT);

    cv::imwrite("../images/original_gray.jpg", gray);
    cv::imwrite("../images/entropy_map.jpg", entropyColor);

    // Side-by-side comparison
    cv::Mat grayBGR;
    cv::cvtColor(gray, grayBGR, cv::COLOR_GRAY2BGR);
    cv::Mat comparison;
    cv::hconcat(grayBGR, entropyColor, comparison);
    cv::imwrite("../images/comparison.jpg", comparison);

    std::cout << "Done! Check entropy_map.jpg and comparison.jpg" << std::endl;
    return 0;
}