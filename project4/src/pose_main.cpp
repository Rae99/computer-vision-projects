#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "target_detection.h"
#include "calibration_utils.h"
#include "pose_estimation.h"

int main() {
    std::cout << "Current working directory: "
              << std::filesystem::current_path() << std::endl;

    std::string calibrationFile = "../data/calibration_results/intrinsics.yml";

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    bool calibrationLoaded = readCalibrationFromFile(
        calibrationFile,
        cameraMatrix,
        distCoeffs
    );

    if (!calibrationLoaded) {
        std::cerr << "Failed to load calibration parameters." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera." << std::endl;
        return -1;
    }

    cv::Size patternSize(9, 6);
    std::vector<cv::Vec3f> worldPoints = buildChessboardWorldPoints(patternSize);

    cv::Mat frame;
    cv::Mat outputFrame;
    std::vector<cv::Point2f> corners;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: empty frame." << std::endl;
            break;
        }

        bool found = detectChessboardCorners(frame, patternSize, corners, outputFrame);

        if (found) {
            cv::Mat rvec;
            cv::Mat tvec;

            bool poseFound = estimateBoardPose(
                worldPoints,
                corners,
                cameraMatrix,
                distCoeffs,
                rvec,
                tvec
            );

            if (poseFound) {
                cv::putText(outputFrame, "Pose estimated", cv::Point(20, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

                std::cout << "\nrvec:\n" << rvec << std::endl;
                std::cout << "tvec:\n" << tvec << std::endl;
            } else {
                cv::putText(outputFrame, "solvePnP failed", cv::Point(20, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
        } else {
            cv::putText(outputFrame, "Checkerboard not found", cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        cv::putText(outputFrame, "Press q to quit", cv::Point(20, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Project 4 - Task 4 Pose Estimation", outputFrame);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}