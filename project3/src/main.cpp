#include <opencv2/opencv.hpp>
#include <iostream>
#include "p3_segmentation.hpp"

int main()
{
    // Open default webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera.\n";
        return -1;
    }

    std::cout << "Project3 Task1 Running...\n";
    std::cout << "Press 'q' to quit.\n";

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // ---- Task 1 ----
        cv::Mat binary = p3::thresholdBinary(frame);

        // Display results
        cv::imshow("Original", frame);
        cv::imshow("Task1 Binary", binary);

        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    return 0;
}