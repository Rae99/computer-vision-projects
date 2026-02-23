#include "p3_segmentation.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

static void printUsage(const char *prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " --cam <index>\n"
              << "  " << prog << " --video <path>\n"
              << "  " << prog << " --dir <folder>\n"
              << "  " << prog << " --image <path>\n\n"
              << "Keys:\n"
              << "  q / ESC : quit\n"
              << "  space   : pause/resume (cam/video)\n"
              << "  n       : next image (dir)\n";
}

static bool isImageFile(const fs::path &p) {
    if (!p.has_extension())
        return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
            ext == ".tiff" || ext == ".tif");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    std::string arg = argv[2];

    bool paused = false;

    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::namedWindow("Task1 Binary", cv::WINDOW_NORMAL);

    // ----------------------
    // Mode: single image
    // ----------------------
    if (mode == "--image") {
        cv::Mat frame = cv::imread(arg, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Failed to read image: " << arg << "\n";
            return 1;
        }

        cv::Mat binary = p3::thresholdBinary(frame);

        cv::imshow("Input", frame);
        cv::imshow("Task1 Binary", binary);
        cv::waitKey(0);
        return 0;
    }

    // ----------------------
    // Mode: directory images
    // ----------------------
    if (mode == "--dir") {
        fs::path folder(arg);
        if (!fs::exists(folder) || !fs::is_directory(folder)) {
            std::cerr << "Not a directory: " << arg << "\n";
            return 1;
        }

        std::vector<fs::path> files;
        for (auto &e : fs::directory_iterator(folder)) {
            if (e.is_regular_file() && isImageFile(e.path())) {
                files.push_back(e.path());
            }
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            std::cerr << "No image files found in: " << arg << "\n";
            return 1;
        }

        size_t idx = 0;
        while (true) {
            cv::Mat frame = cv::imread(files[idx].string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "Failed to read: " << files[idx] << "\n";
                return 1;
            }

            cv::Mat binary = p3::thresholdBinary(frame);

            cv::imshow("Input", frame);
            cv::imshow("Task1 Binary", binary);

            int key = cv::waitKey(0);
            if (key == 27 || key == 'q')
                break;
            if (key == 'n' || key == ' ') {
                idx = (idx + 1) % files.size();
            }
        }
        return 0;
    }

    // ----------------------
    // Mode: video / cam
    // ----------------------
    cv::VideoCapture cap;
    if (mode == "--video") {
        cap.open(arg);
    } else if (mode == "--cam") {
        int camIndex = std::stoi(arg);
        cap.open(camIndex);
    } else {
        printUsage(argv[0]);
        return 1;
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open source: " << mode << " " << arg << "\n";
        return 1;
    }

    while (true) {
        cv::Mat frame;
        if (!paused) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "End of stream or read failed.\n";
                break;
            }
        }

        if (!frame.empty()) {
            cv::Mat binary = p3::thresholdBinary(frame);
            cv::imshow("Input", frame);
            cv::imshow("Task1 Binary", binary);
        }

        int key = cv::waitKey(10);
        if (key == 27 || key == 'q')
            break;
        if (key == ' ')
            paused = !paused;
    }

    return 0;
}