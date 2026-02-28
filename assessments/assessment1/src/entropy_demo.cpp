// compile: g++ entropy_demo.cpp -o entropy_demo `pkg-config --cflags --libs
// opencv4`

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

// Compute Shannon entropy of a grayscale image
double computeEntropy(const cv::Mat &grayImg) {
    // Step 1: compute histogram (256 bins, range 0-255)
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = range;
    cv::Mat hist;
    cv::calcHist(&grayImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Step 2: normalize to get probabilities (sum = 1)
    hist /= (float)grayImg.total();

    // Step 3: H = -sum(p * log2(p)), skip p=0 (0*log0 defined as 0)
    double entropy = 0.0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 0)
            entropy -= p * log2(p);
    }
    return entropy;
}

cv::Mat drawHistogram(const cv::Mat &grayImg, int width = 256,
                      int height = 150) {
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = range;
    cv::Mat hist;
    cv::calcHist(&grayImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    double minVal, maxVal;
    cv::minMaxLoc(hist, &minVal, &maxVal);

    cv::Mat histImg(height, width, CV_8UC1, cv::Scalar(255));

    double scale = (maxVal > 0) ? (height - 1) / maxVal : 0;

    for (int i = 0; i < histSize; i++) {
        int val = cvRound(hist.at<float>(i) * scale);
        cv::line(histImg, cv::Point(i, height - 1),
                 cv::Point(i, height - 1 - val), cv::Scalar(0), 1);
    }
    return histImg;
}

int main() {
    int W = 256, H = 256;

    // --- Image 1: Pure white ---
    cv::Mat white(H, W, CV_8UC1, cv::Scalar(255));

    // --- Image 2: Linear gradient (left=0, right=255) ---
    cv::Mat gradient(H, W, CV_8UC1);
    for (int col = 0; col < W; col++)
        gradient.col(col).setTo(cv::Scalar(col));

    // --- Image 3: Random noise ---
    cv::Mat noise(H, W, CV_8UC1);
    cv::randu(noise, 0, 256);

    // Print entropy values
    std::cout << "White   entropy: " << computeEntropy(white) << std::endl;
    std::cout << "Gradient entropy: " << computeEntropy(gradient) << std::endl;
    std::cout << "Noise   entropy: " << computeEntropy(noise) << std::endl;

    // Save images and histograms to workspace images folder
    cv::imwrite("../images/white.png", white);
    cv::imwrite("../images/gradient.png", gradient);
    cv::imwrite("../images/noise.png", noise);
    cv::imwrite("../images/hist_white.png", drawHistogram(white));
    cv::imwrite("../images/hist_gradient.png", drawHistogram(gradient));
    cv::imwrite("../images/hist_noise.png", drawHistogram(noise));

    return 0;
}

// Expected output (approximate):
// White    entropy: 0.0
// Gradient entropy: ~8.0
// Noise    entropy: ~7.99