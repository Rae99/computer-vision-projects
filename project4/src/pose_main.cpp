/*
 * CS5330 Mar 2026
 * Author: Junyao Han, Junrui Ding
 * Project4-pose_main.cpp
 * Runs Project 4 AR tasks for checkerboard pose estimation and rendering.
 * It draws axes, outer corners, a 3D church model, and a directional arrow
 * sign with support for both live camera mode and static image mode.
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "calibration_utils.h"
#include "pose_estimation.h"
#include "target_detection.h"

using Mat = cv::Mat;
using Pt2 = cv::Point2f;
using V3f = cv::Vec3f;
using Scl = cv::Scalar;

/**
 * @brief Draw XYZ axes at the checkerboard origin.
 */
void drawAxes(Mat &frame, const Mat &rvec, const Mat &tvec, const Mat &K,
              const Mat &D) {
    std::vector<V3f> pts = {{0, 0, 0}, {2, 0, 0}, {0, -2, 0}, {0, 0, 2}};
    std::vector<Pt2> p;
    cv::projectPoints(pts, rvec, tvec, K, D, p);
    cv::arrowedLine(frame, p[0], p[1], Scl(255, 0, 0), 3, 8, 0, 0.15); // X blue
    cv::arrowedLine(frame, p[0], p[2], Scl(0, 200, 0), 3, 8, 0,
                    0.15); // Y green
    cv::arrowedLine(frame, p[0], p[3], Scl(0, 0, 255), 3, 8, 0, 0.15); // Z red
    cv::putText(frame, "X", p[1] + Pt2(6, 0), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                Scl(255, 0, 0), 2);
    cv::putText(frame, "Y", p[2] + Pt2(6, 0), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                Scl(0, 200, 0), 2);
    cv::putText(frame, "Z", p[3] + Pt2(6, 0), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                Scl(0, 0, 255), 2);
}

/**
 * @brief Task 5b: yellow rectangle around the 4 outer corners of the target
 * 9x6 internal corners -> board spans X=[0,8], Y=[0,-5]
 */
void drawOuterCorners(Mat &frame, const Mat &rvec, const Mat &tvec,
                      const Mat &K, const Mat &D) {
    std::vector<V3f> pts = {{0, 0, 0}, {8, 0, 0}, {8, -5, 0}, {0, -5, 0}};
    std::vector<Pt2> p;
    cv::projectPoints(pts, rvec, tvec, K, D, p);
    Scl yellow(0, 255, 255);
    for (int i = 0; i < 4; ++i) {
        cv::line(frame, p[i], p[(i + 1) % 4], yellow, 2);
        cv::circle(frame, p[i], 7, yellow, -1);
    }
}

/**
 * @brief Task 6: Church
 *
 * World-space layout (units = checkerboard squares):
 *   Board spans X=[0,8], Y=[0,-5], Z=0
 *   Nave:       X=[2,6],   Y=[-1,-4.5], Z=0..2.5, gable ridge Z=4.2
 *   Bell tower: X=[2,3.2], Y=[-1,-2.2], Z=0..4.5, spire peak  Z=6.0
 *   Cross:      above spire, Z=6.0..7.4
 */
void drawChurch(Mat &frame, const Mat &rvec, const Mat &tvec, const Mat &K,
                const Mat &D) {
    std::vector<V3f> pts = {
        // Nave floor ring (z=0.2) ----------------------------------------- 0-3
        {2.0f, -1.0f, 0.2f}, // 0  front-left
        {6.0f, -1.0f, 0.2f}, // 1  front-right
        {6.0f, -4.5f, 0.2f}, // 2  back-right
        {2.0f, -4.5f, 0.2f}, // 3  back-left
        // Nave wall top (z=2.5) ------------------------------------------- 4-7
        {2.0f, -1.0f, 2.5f}, // 4
        {6.0f, -1.0f, 2.5f}, // 5
        {6.0f, -4.5f, 2.5f}, // 6
        {2.0f, -4.5f, 2.5f}, // 7
        // Gable roof ridge -------------------------------------------------
        // 8-9
        {4.0f, -1.0f, 4.2f}, // 8  front ridge
        {4.0f, -4.5f, 4.2f}, // 9  back ridge
        // Bell tower extra base corners (z=0.2), shares pt 0 -------------
        // 10-12
        {3.2f, -1.0f, 0.2f}, // 10
        {3.2f, -2.2f, 0.2f}, // 11
        {2.0f, -2.2f, 0.2f}, // 12
        // Bell tower top ring (z=4.5) -------------------------------------
        // 13-16
        {2.0f, -1.0f, 4.5f}, // 13  above pt 0
        {3.2f, -1.0f, 4.5f}, // 14  above pt 10
        {3.2f, -2.2f, 4.5f}, // 15  above pt 11
        {2.0f, -2.2f, 4.5f}, // 16  above pt 12
        // Tower spire peak ------------------------------------------------- 17
        {2.6f, -1.6f, 6.0f}, // 17
        // Cross (on spire) ------------------------------------------------
        // 18-20
        {2.6f, -1.6f, 7.4f}, // 18  cross top
        {2.0f, -1.6f, 6.8f}, // 19  cross left  (X direction)
        {3.2f, -1.6f, 6.8f}, // 20  cross right
        // Door arch (front face y=-1) -------------------------------------
        // 21-23
        {3.5f, -1.0f, 0.2f}, // 21  door bottom-left
        {4.5f, -1.0f, 0.2f}, // 22  door bottom-right
        {4.0f, -1.0f, 2.0f}, // 23  door arch peak
        // Side window (right wall x=6) ------------------------------------
        // 24-28
        {6.0f, -2.0f, 1.0f}, // 24
        {6.0f, -3.0f, 1.0f}, // 25
        {6.0f, -3.0f, 2.2f}, // 26
        {6.0f, -2.0f, 2.2f}, // 27
        {6.0f, -2.5f, 2.5f}, // 28  arch peak
    };

    std::vector<Pt2> p;
    cv::projectPoints(pts, rvec, tvec, K, D, p);

    auto L = [&](int a, int b, Scl c, int t = 2) {
        cv::line(frame, p[a], p[b], c, t);
    };

    Scl gray(170, 170, 170);
    Scl cyan(200, 210, 0);
    Scl red(0, 50, 255);
    Scl yellow(0, 230, 230);
    Scl white(220, 220, 220);

    // Nave
    L(0, 1, gray);
    L(1, 2, gray);
    L(2, 3, gray);
    L(3, 0, gray);
    L(0, 4, gray, 3);
    L(1, 5, gray, 3);
    L(2, 6, gray, 3);
    L(3, 7, gray, 3);
    L(4, 5, gray);
    L(5, 6, gray);
    L(6, 7, gray);
    L(7, 4, gray);
    // Gable
    L(4, 8, red, 3);
    L(5, 8, red, 3);
    L(7, 9, red, 3);
    L(6, 9, red, 3);
    L(8, 9, red, 3);
    // Tower base
    L(0, 10, cyan);
    L(10, 11, cyan);
    L(11, 12, cyan);
    L(12, 0, cyan);
    // Tower pillars
    L(0, 13, cyan, 3);
    L(10, 14, cyan, 3);
    L(11, 15, cyan, 3);
    L(12, 16, cyan, 3);
    // Tower top ring
    L(13, 14, cyan);
    L(14, 15, cyan);
    L(15, 16, cyan);
    L(16, 13, cyan);
    // Tower spire
    L(13, 17, red, 2);
    L(14, 17, red, 2);
    L(15, 17, red, 2);
    L(16, 17, red, 2);
    // Cross
    L(17, 18, yellow, 3);
    L(19, 20, yellow, 3);
    // Door arch
    L(21, 22, white);
    L(21, 23, white);
    L(22, 23, white);
    // Side window + arch
    L(24, 25, white);
    L(25, 26, white);
    L(26, 27, white);
    L(27, 24, white);
    L(27, 28, white);
    L(26, 28, white);
}

/**
 * @brief Extension 1: arrow sign on board B pointing toward board A (church).
 * @details
 * Logic:
 *   1. tvec_church - tvec_sign  = direction in camera space
 *   2. R_sign^T * direction     = direction in board B's local frame
 *   3. atan2(dy, dx)            = in-plane angle on board B
 *   4. Rotate arrow vertices by that angle around Z
 */
void drawArrowSign(Mat &frame, const Mat &rvec_sign, const Mat &tvec_sign,
                   const Mat &tvec_church, bool churchVisible, const Mat &K,
                   const Mat &D) {
    double angle = 0.0;
    if (churchVisible && !tvec_church.empty()) {
        Mat dir_cam = tvec_church - tvec_sign; // 3x1 in camera space
        Mat R;
        cv::Rodrigues(rvec_sign, R);
        Mat dir_local = R.t() * dir_cam; // board B local frame
        angle = std::atan2(dir_local.at<double>(1), dir_local.at<double>(0));
    }

    const float cx = 2.5f, cy_w = -1.5f; // sign center on board B (world XY)
    const float pole_h = 2.8f;           // Z of arrow plane

    auto rotXY = [&](float x, float y) -> std::pair<float, float> {
        float ca = (float)std::cos(angle), sa = (float)std::sin(angle);
        return {x * ca - y * sa + cx, x * sa + y * ca + cy_w};
    };

    std::vector<V3f> pts;

    // Pole bottom / top -------------------------------------------------- 0,1
    pts.push_back({cx, cy_w, 0.2f});
    pts.push_back({cx, cy_w, pole_h});

    // Arrow shaft -------------------------------------------------------- 2,3
    auto [sx, sy] = rotXY(-0.9f, 0.0f);
    pts.push_back({sx, sy, pole_h}); // 2 shaft start

    auto [bx, by] = rotXY(0.3f, 0.0f);
    pts.push_back({bx, by, pole_h}); // 3 head base

    // Arrow head ---------------------------------------------------------
    // 4,5,6
    auto [lx, ly] = rotXY(0.3f, -0.45f);
    pts.push_back({lx, ly, pole_h}); // 4 head left

    auto [tx, ty] = rotXY(1.1f, 0.0f);
    pts.push_back({tx, ty, pole_h}); // 5 tip

    auto [rx, ry] = rotXY(0.3f, 0.45f);
    pts.push_back({rx, ry, pole_h}); // 6 head right

    // Sign board (vertical rectangle along arrow direction) ------------- 7-10
    const float hw = 1.1f, hz = 0.45f;
    auto [bl_x, bl_y] = rotXY(-hw, 0.0f);
    auto [br_x, br_y] = rotXY(hw, 0.0f);
    pts.push_back({bl_x, bl_y, pole_h - hz}); // 7
    pts.push_back({br_x, br_y, pole_h - hz}); // 8
    pts.push_back({br_x, br_y, pole_h + hz}); // 9
    pts.push_back({bl_x, bl_y, pole_h + hz}); // 10

    std::vector<Pt2> p;
    cv::projectPoints(pts, rvec_sign, tvec_sign, K, D, p);

    auto L = [&](int a, int b, Scl c, int t = 2) {
        cv::line(frame, p[a], p[b], c, t);
    };

    Scl white(220, 220, 220);
    Scl orange(0, 140, 255); // BGR -> orange

    L(0, 1, white, 3); // pole
    L(7, 8, white);
    L(8, 9, white);
    L(9, 10, white);
    L(10, 7, white);    // sign board
    L(2, 3, orange, 3); // shaft
    L(3, 4, orange, 2);
    L(4, 5, orange, 3); // head left side + tip
    L(5, 6, orange, 3);
    L(6, 3, orange, 2); // tip + head right side
}

/**
 * @brief Entry point for AR pose estimation in live or static-image mode.
 */
int main(int argc, char *argv[]) {
    std::cout << "Working dir: " << std::filesystem::current_path() << "\n";

    // Extension 2: static image mode
    const bool staticMode = (argc > 1);
    const std::string imagePath = staticMode ? argv[1] : "";

    // Load calibration
    const std::string calibFile = "../data/calibration_results/intrinsics.yml";
    Mat K, D;
    if (!readCalibrationFromFile(calibFile, K, D)) {
        std::cerr << "Cannot load calibration: " << calibFile << "\n";
        return -1;
    }

    // Pattern sizes
    //   Board A (9x6): church          <- same board used for calibration
    //   Board B (5x4): arrow sign      <- needs a separate 5x4 checkerboard
    cv::Size patternA(9, 6);
    cv::Size patternB(5, 4);
    std::vector<V3f> worldA = buildChessboardWorldPoints(patternA);
    std::vector<V3f> worldB = buildChessboardWorldPoints(patternB);

    // Open camera or load static image
    cv::VideoCapture cap;
    Mat staticFrame;

    if (staticMode) {
        staticFrame = cv::imread(imagePath);
        if (staticFrame.empty()) {
            std::cerr << "Cannot load image: " << imagePath << "\n";
            return -1;
        }
        std::cout << "[Static image] " << imagePath << "\n";
    } else {
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open camera\n";
            return -1;
        }
        std::cout << "Live mode\n"
                  << "Board A (9x6) = church    Board B (5x4) = arrow sign\n"
                  << "q = quit    s = screenshot\n";
    }

    int shotCount = 0;

    while (true) {
        Mat frame;
        if (staticMode) {
            staticFrame.copyTo(frame);
        } else {
            cap >> frame;
            if (frame.empty())
                break;
        }

        // Keep a clean copy for detection (so board A drawings don't confuse
        // board B)
        Mat cleanFrame;
        frame.copyTo(cleanFrame);

        // --- Board A: church ---------------------------------------------
        std::vector<Pt2> cornersA;
        Mat outA;
        bool foundA =
            detectChessboardCorners(cleanFrame, patternA, cornersA, outA);
        frame = outA; // show board A corner overlay

        Mat rvecA, tvecA;
        bool poseA = false;

        if (foundA) {
            poseA = estimateBoardPose(worldA, cornersA, K, D, rvecA, tvecA);
            if (poseA) {
                drawOuterCorners(frame, rvecA, tvecA, K, D);
                drawAxes(frame, rvecA, tvecA, K, D);
                drawChurch(frame, rvecA, tvecA, K, D);
                cv::putText(frame, "Church OK", cv::Point(20, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, Scl(0, 255, 0), 2);
                std::cout << "\rrvec[" << rvecA.at<double>(0) << " "
                          << rvecA.at<double>(1) << " " << rvecA.at<double>(2)
                          << "] tvec[" << tvecA.at<double>(0) << " "
                          << tvecA.at<double>(1) << " " << tvecA.at<double>(2)
                          << "]   " << std::flush;
            }
        } else {
            cv::putText(frame, "Board A not found", cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, Scl(0, 0, 255), 2);
        }

        // --- Board B: arrow sign (live mode only) ------------------------
        if (!staticMode) {
            std::vector<Pt2> cornersB;
            Mat outB_unused;
            // Detect on cleanFrame so board A drawings don't interfere
            bool foundB = detectChessboardCorners(cleanFrame, patternB,
                                                  cornersB, outB_unused);
            if (foundB) {
                Mat rvecB, tvecB;
                bool poseB =
                    estimateBoardPose(worldB, cornersB, K, D, rvecB, tvecB);
                if (poseB) {
                    drawArrowSign(frame, rvecB, tvecB, poseA ? tvecA : Mat(),
                                  poseA, K, D);
                    cv::putText(frame, "Arrow OK", cv::Point(20, 65),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, Scl(0, 220, 255),
                                2);
                }
            }
        }

        cv::putText(frame,
                    staticMode ? "press any key to quit"
                               : "q=quit  s=screenshot",
                    cv::Point(20, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, Scl(180, 180, 180), 1);

        cv::imshow("Project 4 - AR", frame);

        char key = (char)cv::waitKey(staticMode ? 0 : 10);
        if (key == 'q' || staticMode)
            break;
        if (key == 's') {
            std::filesystem::create_directories("../data/screenshots");
            std::string fname = "../data/screenshots/ar_" +
                                std::to_string(++shotCount) + ".png";
            cv::imwrite(fname, frame);
            std::cout << "\nSaved: " << fname << "\n";
        }
    }

    std::cout << "\n";
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
