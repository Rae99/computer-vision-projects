#include "p3_segmentation.hpp"
#include <cmath>

namespace p3 {

cv::Mat thresholdBinary(const cv::Mat& bgr)
{
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // Smooth noise
    cv::Mat smooth;
    cv::GaussianBlur(gray, smooth, cv::Size(5,5), 0);

    // ---- ISODATA thresholding ----
    double T = 128.0;
    double prevT = 0.0;

    while (std::abs(T - prevT) > 0.5)
    {
        prevT = T;

        double sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;

        for (int y = 0; y < smooth.rows; ++y)
        {
            const uchar* row = smooth.ptr<uchar>(y);
            for (int x = 0; x < smooth.cols; ++x)
            {
                if (row[x] < T) { sum1 += row[x]; count1++; }
                else            { sum2 += row[x]; count2++; }
            }
        }

        double m1 = (count1 > 0) ? sum1 / count1 : 0;
        double m2 = (count2 > 0) ? sum2 / count2 : 0;

        T = (m1 + m2) / 2.0;
    }

    cv::Mat binary;
    cv::threshold(smooth, binary, T, 255, cv::THRESH_BINARY);

    return binary;
}

}