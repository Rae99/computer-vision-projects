#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "calibration_utils.h"
#include "pose_estimation.h"
#include "target_detection.h"

// ============================================================
// Task 5: Draw 3D coordinate axes at the board origin
//   X axis = blue  (right along first row)
//   Y axis = green (down along first column)
//   Z axis = red   (toward the viewer)
// ============================================================
void drawAxes(cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec,
              const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) {
    // Axis tip positions in world space (2 squares long each)
    std::vector<cv::Vec3f> axisPoints = {
        {0.0f, 0.0f, 0.0f},  // 0: origin
        {2.0f, 0.0f, 0.0f},  // 1: +X tip
        {0.0f, -2.0f, 0.0f}, // 2: +Y tip  (Y is negative in world coords)
        {0.0f, 0.0f, 2.0f}   // 3: +Z tip  (toward viewer)
    };

    std::vector<cv::Point2f> projected;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs,
                      projected);

    cv::Point2f origin = projected[0];

    cv::arrowedLine(frame, origin, projected[1], cv::Scalar(255, 0, 0), 3, 8, 0,
                    0.15); // X = blue
    cv::arrowedLine(frame, origin, projected[2], cv::Scalar(0, 200, 0), 3, 8, 0,
                    0.15); // Y = green
    cv::arrowedLine(frame, origin, projected[3], cv::Scalar(0, 0, 255), 3, 8, 0,
                    0.15); // Z = red

    cv::putText(frame, "X", projected[1] + cv::Point2f(6, 0),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 0, 0), 2);
    cv::putText(frame, "Y", projected[2] + cv::Point2f(6, 0),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 200, 0), 2);
    cv::putText(frame, "Z", projected[3] + cv::Point2f(6, 0),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);
}

// ============================================================
// Task 5: Project and draw the 4 outer corners of the board
//   9x6 internal corners → board spans X=[0,8], Y=[0,-5]
// ============================================================
void drawOuterCorners(cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec,
                      const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) {
    std::vector<cv::Vec3f> corners3D = {
        {0.0f, 0.0f, 0.0f},  // top-left
        {8.0f, 0.0f, 0.0f},  // top-right
        {8.0f, -5.0f, 0.0f}, // bottom-right
        {0.0f, -5.0f, 0.0f}  // bottom-left
    };

    std::vector<cv::Point2f> projected;
    cv::projectPoints(corners3D, rvec, tvec, cameraMatrix, distCoeffs,
                      projected);

    cv::Scalar color(0, 255, 255); // yellow
    for (int i = 0; i < 4; i++) {
        cv::line(frame, projected[i], projected[(i + 1) % 4], color, 2);
        cv::circle(frame, projected[i], 7, color, -1);
    }
}

// ============================================================
// Task 6: Project and draw a virtual house floating above the board
//
// Parts and colors (BGR):
//   Green   (0,220,0)   – floor ring
//   Cyan    (220,210,0) – walls (pillars + ceiling ring)
//   Red     (0,50,255)  – gable roof
//   Magenta (220,0,200) – chimney
//
// World space layout (units = checkerboard squares):
//   Board:  X in [0,8], Y in [-5,0], Z=0
//   House centred at (4,-2.5), floats Z=0.3..4.0
// ============================================================
void drawVirtualHouse(cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec,
                      const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs) {
    std::vector<cv::Vec3f> pts = {
        // Floor ring  (z = 0.3)
        {2.5f, -1.0f, 0.3f}, //  0  front-left
        {5.5f, -1.0f, 0.3f}, //  1  front-right
        {5.5f, -4.0f, 0.3f}, //  2  back-right
        {2.5f, -4.0f, 0.3f}, //  3  back-left

        // Ceiling ring  (z = 2.5)
        {2.5f, -1.0f, 2.5f}, //  4  front-left
        {5.5f, -1.0f, 2.5f}, //  5  front-right
        {5.5f, -4.0f, 2.5f}, //  6  back-right
        {2.5f, -4.0f, 2.5f}, //  7  back-left

        // Gable roof ridge (z = 4.0, centred Y=-2.5)
        {2.5f, -2.5f, 4.0f}, //  8  left  ridge end
        {5.5f, -2.5f, 4.0f}, //  9  right ridge end

        // Chimney base on ceiling (near front-right)
        {4.8f, -1.3f, 2.5f}, // 10
        {5.3f, -1.3f, 2.5f}, // 11
        {5.3f, -1.8f, 2.5f}, // 12
        {4.8f, -1.8f, 2.5f}, // 13

        // Chimney top  (z = 3.4)
        {4.8f, -1.3f, 3.4f}, // 14
        {5.3f, -1.3f, 3.4f}, // 15
        {5.3f, -1.8f, 3.4f}, // 16
        {4.8f, -1.8f, 3.4f}, // 17
    };

    std::vector<cv::Point2f> p;
    cv::projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs, p);

    // Helper: draw line by vertex index
    auto L = [&](int a, int b, cv::Scalar color, int thickness = 2) {
        cv::line(frame, p[a], p[b], color, thickness);
    };

    cv::Scalar green(0, 220, 0);
    cv::Scalar cyan(220, 210, 0);
    cv::Scalar red(0, 50, 255);
    cv::Scalar magenta(220, 0, 200);

    // Floor ring
    L(0, 1, green);
    L(1, 2, green);
    L(2, 3, green);
    L(3, 0, green);

    // Four vertical pillars
    L(0, 4, cyan, 3);
    L(1, 5, cyan, 3);
    L(2, 6, cyan, 3);
    L(3, 7, cyan, 3);

    // Ceiling ring
    L(4, 5, cyan);
    L(5, 6, cyan);
    L(6, 7, cyan);
    L(7, 4, cyan);

    // Gable roof
    L(4, 8, red, 3);
    L(7, 8, red, 3); // left  gable triangle
    L(5, 9, red, 3);
    L(6, 9, red, 3); // right gable triangle
    L(8, 9, red, 3); // ridge beam

    // Chimney base square
    L(10, 11, magenta);
    L(11, 12, magenta);
    L(12, 13, magenta);
    L(13, 10, magenta);
    // Chimney top square
    L(14, 15, magenta);
    L(15, 16, magenta);
    L(16, 17, magenta);
    L(17, 14, magenta);
    // Chimney vertical edges
    L(10, 14, magenta);
    L(11, 15, magenta);
    L(12, 16, magenta);
    L(13, 17, magenta);
}

int main() {
    std::cout << "Current working directory: "
              << std::filesystem::current_path() << std::endl;

    const std::string calibrationFile =
        "../data/calibration_results/intrinsics.yml";

    cv::Mat cameraMatrix, distCoeffs;
    if (!readCalibrationFromFile(calibrationFile, cameraMatrix, distCoeffs)) {
        std::cerr << "Failed to load calibration parameters from: "
                  << calibrationFile << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera." << std::endl;
        return -1;
    }

    cv::Size patternSize(9, 6);
    std::vector<cv::Vec3f> worldPoints =
        buildChessboardWorldPoints(patternSize);

    cv::Mat frame, outputFrame;
    std::vector<cv::Point2f> corners;
    int screenshotCount = 0;

    std::cout << "Controls:  q = quit     s = save screenshot" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: empty frame." << std::endl;
            break;
        }

        bool found =
            detectChessboardCorners(frame, patternSize, corners, outputFrame);

        if (found) {
            cv::Mat rvec, tvec;
            bool poseFound = estimateBoardPose(
                worldPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            if (poseFound) {
                // ---- Task 5: axes + outer corner projection ----
                drawOuterCorners(outputFrame, rvec, tvec, cameraMatrix,
                                 distCoeffs);
                drawAxes(outputFrame, rvec, tvec, cameraMatrix, distCoeffs);

                // ---- Task 6: virtual house AR object ----
                drawVirtualHouse(outputFrame, rvec, tvec, cameraMatrix,
                                 distCoeffs);

                // Print pose on one line so the terminal doesn't flood
                std::cout << "\rrvec: [" << rvec.at<double>(0) << ", "
                          << rvec.at<double>(1) << ", " << rvec.at<double>(2)
                          << "]  tvec: [" << tvec.at<double>(0) << ", "
                          << tvec.at<double>(1) << ", " << tvec.at<double>(2)
                          << "]      " << std::flush;

                cv::putText(outputFrame, "Pose OK", cv::Point(20, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 255, 0), 2);
            } else {
                cv::putText(outputFrame, "solvePnP failed", cv::Point(20, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 0, 255), 2);
            }
        } else {
            cv::putText(outputFrame, "Board not found", cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255),
                        2);
        }

        cv::putText(outputFrame, "q=quit  s=screenshot", cv::Point(20, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200),
                    1);

        cv::imshow("Project 4 - AR (Tasks 5+6)", outputFrame);

        char key = static_cast<char>(cv::waitKey(10));
        if (key == 'q')
            break;
        if (key == 's') {
            std::filesystem::create_directories("../data/screenshots");
            std::string fname = "../data/screenshots/ar_" +
                                std::to_string(++screenshotCount) + ".png";
            cv::imwrite(fname, outputFrame);
            std::cout << "\nSaved screenshot: " << fname << std::endl;
        }
    }

    std::cout << std::endl;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}