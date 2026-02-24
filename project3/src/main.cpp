#include "p3_db.hpp"
#include "p3_segmentation.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool isImageFile(const fs::path &p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
            ext == ".tiff");
}

static void showUsage() {
    std::cerr << "Usage:\n"
              << "  ./main --task1 --image path/to/img.jpg\n"
              << "  ./main --task2 --image path/to/img.jpg\n"
              << "  ./main --task3 --image path/to/img.jpg\n"
              << "  ./main --task4 --image path/to/img.jpg\n"
              << "  ./main --task5 --image path/to/img.jpg\n"
              << "  ./main --task6 --image path/to/img.jpg\n"
              << "  ./main --task1 --dir   path/to/folder\n"
              << "  ./main --task2 --dir   path/to/folder\n"
              << "  ./main --task3 --dir   path/to/folder\n"
              << "  ./main --task4 --dir   path/to/folder\n"
              << "  ./main --task5 --dir   path/to/folder\n"
              << "  ./main --task6 --dir   path/to/folder\n";
}

struct Task4Context {
    cv::Mat labels;
    std::vector<int> regionIds;
    std::vector<p3::RegionFeatures> features;
};

static Task4Context computeMajorRegionFeatures(const cv::Mat &cleaned,
                                               int connectivity,
                                               int minArea,
                                               int maxRegions) {
    Task4Context ctx;

    cv::Mat stats, centroids;
    int n = cv::connectedComponentsWithStats(cleaned, ctx.labels, stats, centroids,
                                             connectivity, CV_32S);
    if (n <= 1) {
        return ctx;
    }

    ctx.regionIds = p3::selectRegionsByScore(stats, centroids,
                                             cleaned.cols, cleaned.rows,
                                             minArea, maxRegions,
                                             0.002, true);

    for (int id : ctx.regionIds) {
        ctx.features.push_back(p3::computeRegionFeatures(ctx.labels, id));
    }

    return ctx;
}

static void runOnFrame(const cv::Mat &frame,
                       const std::string &task,
                       const std::string &dbPath) {
    (void)dbPath;

    cv::Mat binary1, cleaned;

    if (task == "--task1") {
        binary1 = p3::thresholdBinary(frame);
        cv::imshow("Input", frame);
        cv::imshow("Task1 Binary", binary1);
        return;
    }

    if (task == "--task2") {
        binary1 = p3::thresholdBinary(frame);
        cleaned = p3::morphCleanup(binary1, 3, 15);

        cv::imshow("Input", frame);
        cv::imshow("Task1 Binary", binary1);
        cv::imshow("Task2 Cleaned", cleaned);
        return;
    }

    if (task == "--task3") {
        binary1 = p3::thresholdBinary(frame);
        cleaned = p3::morphCleanup(binary1, 3, 15);

        cv::Mat regions = p3::regionMapColor(cleaned, 500, 8, 10);

        cv::imshow("Input", frame);
        cv::imshow("Task2 Cleaned", cleaned);
        cv::imshow("Task3 Regions", regions);
        return;
    }

    if (task == "--task4" || task == "--task5" || task == "--task6") {
    binary1 = p3::thresholdBinary(frame);
    cleaned = p3::morphCleanup(binary1, 3, 15);

    int minArea = 500;
    int maxRegions = 1;
    Task4Context ctx = computeMajorRegionFeatures(cleaned, 8, minArea, maxRegions);

    cv::Mat vis = frame.clone();

    // Load DB + stats once per frame is okay for now (simple & correct).
    // If you want faster: cache them outside.
    std::vector<p3::DBSample> samples;
    p3::DBStats stats;
    bool dbOK = false;
    if (task == "--task6") {
        dbOK = p3::loadDB(dbPath, samples);
        if (dbOK) stats = p3::computeDBStats(samples);
    }

    for (size_t i = 0; i < ctx.features.size(); ++i) {
        p3::drawRegionOverlay(vis, ctx.features[i]);

        std::vector<double> fv = ctx.features[i].featureVector();

        // Task4: print feature vector
        if (task == "--task4" || task == "--task5") {
            std::cout << "Region " << ctx.features[i].regionId
                      << " feature vector: [";
            for (size_t k = 0; k < fv.size(); ++k) {
                std::cout << fv[k] << (k + 1 < fv.size() ? ", " : "");
            }
            std::cout << "]\n";
        }

        // Task6: classify + overlay label
        if (task == "--task6") {
            std::string label = "UNKNOWN";
            double bestDist = 0.0;

            if (dbOK) {
                // optional unknown threshold (you can tune)
                double thr = 20.0; // try 10~50 depending on your data
                label = p3::classifyNNWithUnknown(fv, samples, stats, thr, bestDist);
            }

            // draw label near centroid
            cv::Point org((int)std::round(ctx.features[i].centroid.x) + 10,
                          (int)std::round(ctx.features[i].centroid.y) + 25);

            std::string text = label;
            cv::putText(vis, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2);

            std::cout << "Predicted: " << label << " dist=" << bestDist << "\n";
        }
    }

    cv::imshow("Input", frame);
    cv::imshow("Task2 Cleaned", cleaned);
    cv::imshow("Task4/6 Output", vis);
    return;
    }

    std::cerr << "Unknown task: " << task << "\n";
}

static bool saveTrainingSampleFromLastFrame(const cv::Mat &frame,
                                            const std::string &dbPath) {
    cv::Mat binary1 = p3::thresholdBinary(frame);
    cv::Mat cleaned = p3::morphCleanup(binary1, 3, 15);

    int minArea = 500;
    int maxRegions = 1;
    Task4Context ctx = computeMajorRegionFeatures(cleaned, 8, minArea, maxRegions);

    if (ctx.features.empty()) {
        std::cerr << "No valid region found to save.\n";
        return false;
    }

    std::vector<double> fv = ctx.features[0].featureVector();

    std::cout << "Enter label (no spaces): ";
    std::string label;
    std::cin >> label;

    bool ok = p3::appendSample(dbPath, label, fv);
    if (ok) {
        std::cout << "Saved to DB: " << label << " [";
        for (size_t k = 0; k < fv.size(); ++k) {
            std::cout << fv[k] << (k + 1 < fv.size() ? ", " : "");
        }
        std::cout << "] -> " << dbPath << "\n";
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        showUsage();
        return 1;
    }

    std::string task = argv[1];
    std::string mode = argv[2];
    std::string arg = argv[3];

    std::string dbPath = "data/object_db.txt";

    if (mode == "--image") {
        cv::Mat frame = cv::imread(arg, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Failed to read image: " << arg << "\n";
            return 1;
        }

        runOnFrame(frame, task, dbPath);

        while (true) {
            int key = cv::waitKey(0);
            if (key == 27 || key == 'q') break;

            // Task5: press 'N' to save training sample
            if (task == "--task5" && (key == 'N')) {
                saveTrainingSampleFromLastFrame(frame, dbPath);
            }
        }
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

            runOnFrame(frame, task, dbPath);

            int key = cv::waitKey(0);
            if (key == 27 || key == 'q') break;

            // next image
            if (key == 'n' || key == ' ') {
                idx = (idx + 1) % files.size();
                continue;
            }

            // Task5: save training sample (uppercase N)
            if (task == "--task5" && key == 'N') {
                saveTrainingSampleFromLastFrame(frame, dbPath);
            }
        }
        return 0;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    showUsage();
    return 1;
}