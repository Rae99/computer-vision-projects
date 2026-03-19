#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "target_detection.h"
#include "calibration_utils.h"
#include "camera_calibration.h"

int main() {
    std::cout << "Current working directory: "
              << std::filesystem::current_path() << std::endl;

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera." << std::endl;
        return -1;
    }

    cv::Size patternSize(9, 6);

    cv::Mat frame;
    cv::Mat outputFrame;
    std::vector<cv::Point2f> corners;

    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Vec3f>> point_list;

    cv::Mat lastSuccessfulFrame;
    std::vector<cv::Point2f> lastSuccessfulCorners;
    bool hasLastSuccessfulDetection = false;

    std::string calibrationImageFolder = "../data/calibration_images";
    std::string calibrationResultFile = "../data/calibration_results/intrinsics.yml";

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double reprojectionError = -1.0;
    bool calibrationReady = false;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: empty frame." << std::endl;
            break;
        }

        bool found = detectChessboardCorners(frame, patternSize, corners, outputFrame);

        if (found) {
            lastSuccessfulFrame = outputFrame.clone();
            lastSuccessfulCorners = corners;
            hasLastSuccessfulDetection = true;

            std::string line1 = "Checkerboard detected";
            std::string line2 = "Corners: " + std::to_string(corners.size());

            cv::putText(outputFrame, line1, cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            cv::putText(outputFrame, line2, cv::Point(20, 65),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(outputFrame, "Checkerboard not found", cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        std::string savedText = "Saved calibration frames: " + std::to_string(corner_list.size());
        cv::putText(outputFrame, savedText, cv::Point(20, 100),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

        cv::putText(outputFrame, "Press s = save, c = calibrate, w = write file, q = quit",
                    cv::Point(20, 135), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1);

        if (calibrationReady) {
            std::string errorText = "Reprojection error: " + std::to_string(reprojectionError);
            cv::putText(outputFrame, errorText, cv::Point(20, 170),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("Project 4 - Calibration", outputFrame);

        char key = static_cast<char>(cv::waitKey(10));

        if (key == 'q') {
            break;
        } else if (key == 's') {
            if (hasLastSuccessfulDetection) {
                std::vector<cv::Vec3f> point_set = buildChessboardWorldPoints(patternSize);

                if (point_set.size() != lastSuccessfulCorners.size()) {
                    std::cerr << "Error: point_set size does not match corner_set size." << std::endl;
                    continue;
                }

                corner_list.push_back(lastSuccessfulCorners);
                point_list.push_back(point_set);

                bool imageSaved = saveCalibrationImage(
                    lastSuccessfulFrame,
                    calibrationImageFolder,
                    static_cast<int>(corner_list.size())
                );

                std::cout << "\nSaved calibration sample " << corner_list.size() << std::endl;
                std::cout << "corner_list size = " << corner_list.size() << std::endl;
                std::cout << "point_list size = " << point_list.size() << std::endl;
                std::cout << "last corner set size = " << corner_list.back().size() << std::endl;
                std::cout << "last point set size = " << point_list.back().size() << std::endl;

                if (!corner_list.back().empty()) {
                    std::cout << "First 2D corner = ("
                              << corner_list.back()[0].x << ", "
                              << corner_list.back()[0].y << ")" << std::endl;
                }

                if (!point_list.back().empty()) {
                    std::cout << "First 3D point = ("
                              << point_list.back()[0][0] << ", "
                              << point_list.back()[0][1] << ", "
                              << point_list.back()[0][2] << ")" << std::endl;
                }

                if (imageSaved) {
                    std::cout << "Saved calibration image to " << calibrationImageFolder << std::endl;
                } else {
                    std::cout << "Warning: could not save calibration image." << std::endl;
                }
            } else {
                std::cout << "No successful checkerboard detection available to save." << std::endl;
            }
        } else if (key == 'c') {
            if (corner_list.size() < 5) {
                std::cout << "Need at least 5 saved calibration frames before calibrating." << std::endl;
                continue;
            }

            bool success = runCameraCalibration(
                point_list,
                corner_list,
                frame.size(),
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs,
                reprojectionError
            );

            calibrationReady = success;
        } else if (key == 'w') {
            if (!calibrationReady) {
                std::cout << "No calibration available yet. Press c first." << std::endl;
                continue;
            }

            bool success = writeCalibrationToFile(
                calibrationResultFile,
                cameraMatrix,
                distCoeffs,
                reprojectionError
            );

            if (!success) {
                std::cout << "Failed to write calibration file." << std::endl;
            }
        }
    }

    std::cout << "\nFinal summary:" << std::endl;
    std::cout << "Saved corner sets: " << corner_list.size() << std::endl;
    std::cout << "Saved point sets: " << point_list.size() << std::endl;

    if (calibrationReady) {
        std::cout << "Final calibrated camera matrix:\n" << cameraMatrix << std::endl;
        std::cout << "Final distortion coefficients:\n" << distCoeffs << std::endl;
        std::cout << "Final reprojection error: " << reprojectionError << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}