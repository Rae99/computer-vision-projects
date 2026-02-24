#include "p3_segmentation.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace fs = std::filesystem;

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "p3_segmentation.hpp"

namespace fs = std::filesystem;

static bool isImageFile(const fs::path &p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
            ext == ".tiff");
}

static void runOnFrame(const cv::Mat &frame, const std::string &task) {
    cv::Mat binary1, binary2;

    if (task == "--task1") {
        binary1 = p3::thresholdBinary(frame);
        cv::imshow("Input", frame);
        cv::imshow("Task1 Binary", binary1);
    } else if (task == "--task2") {
        binary1 = p3::thresholdBinary(frame);
        binary2 = p3::morphCleanup(binary1, 3, 15);
        binary2 = p3::keepLargestComponent(binary2);

        cv::imshow("Input", frame);
        cv::imshow("Task1 Binary", binary1);
        cv::imshow("Task2 Cleaned", binary2);
    } else {
        std::cerr << "Unknown task: " << task << "\n";
        return;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n"
                  << "  ./main --task1 --image path/to/img.jpg\n"
                  << "  ./main --task2 --image path/to/img.jpg\n"
                  << "  ./main --task1 --dir   path/to/folder\n"
                  << "  ./main --task2 --dir   path/to/folder\n";
        return 1;
    }

    std::string task = argv[1]; // --task1 or --task2 ...
    std::string mode = argv[2]; // --image or --dir
    std::string arg = argv[3];  // path

    if (mode == "--image") {
        cv::Mat frame = cv::imread(arg, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Failed to read image: " << arg << "\n";
            return 1;
        }
        runOnFrame(frame, task);
        cv::waitKey(0);
        return 0;
    }

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
            std::cerr << "No image files in: " << arg << "\n";
            return 1;
        }

        size_t idx = 0;
        while (true) {
            cv::Mat frame = cv::imread(files[idx].string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "Failed to read: " << files[idx] << "\n";
                return 1;
            }

            runOnFrame(frame, task);

            int key = cv::waitKey(0);
            if (key == 27 || key == 'q')
                break;
            if (key == 'n' || key == ' ')
                idx = (idx + 1) % files.size();
        }
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
}