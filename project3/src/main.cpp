#include "p3_db.hpp"
#include "p3_embedding.hpp"
#include "p3_segmentation.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
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
              << "  ./main --task9 --image path/to/img.jpg\n"
              << "  ./main --task1 --dir   path/to/folder\n"
              << "  ./main --task2 --dir   path/to/folder\n"
              << "  ./main --task3 --dir   path/to/folder\n"
              << "  ./main --task4 --dir   path/to/folder\n"
              << "  ./main --task5 --dir   path/to/folder\n"
              << "  ./main --task6 --dir   path/to/folder\n"
              << "  ./main --task9 --dir   path/to/folder\n"
              << "  ./main --task7 --dir   path/to/eval_folder\n";
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

static std::string trueLabelFromFilename(const fs::path &p) {
    std::string stem = p.stem().string();
    size_t pos = stem.find('_');
    if (pos != std::string::npos) {
        return stem.substr(0, pos);
    }
    return stem;
}

static std::string normalizeLabel(const std::string &label) {
    std::string s = label;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    if (s.rfind("pen", 0) == 0) return "pen";
    if (s.rfind("bolt", 0) == 0) return "bolt";
    if (s.rfind("ring", 0) == 0) return "ring";
    if (s.rfind("scissor", 0) == 0) return "scissors";
    if (s.rfind("lock", 0) == 0) return "lock";

    return "UNKNOWN";
}

static bool saveTrainingSampleFromLastFrame(const cv::Mat &frame,
                                            const std::string &dbPath) {
    cv::Mat binary1 = p3::thresholdBinary(frame);
    cv::Mat cleaned = p3::morphCleanup(binary1, 3, 15);

    Task4Context ctx = computeMajorRegionFeatures(cleaned, 8, 500, 1);

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

static bool saveEmbeddingSampleFromLastFrame(const cv::Mat &frame,
                                             const std::string &embedDbPath,
                                             p3::EmbeddingModel &model) {
    cv::Mat binary1 = p3::thresholdBinary(frame);
    cv::Mat cleaned = p3::morphCleanup(binary1, 3, 15);

    Task4Context ctx = computeMajorRegionFeatures(cleaned, 8, 500, 1);
    if (ctx.features.empty()) {
        std::cerr << "No valid region found to save embedding.\n";
        return false;
    }

    std::vector<float> emb = model.computeEmbedding(frame, ctx.features[0]);
    if (emb.empty()) {
        std::cerr << "Embedding empty.\n";
        return false;
    }

    std::cout << "Enter label (no spaces): ";
    std::string label;
    std::cin >> label;

    bool ok = p3::appendEmbeddingSample(embedDbPath, label, emb);
    if (ok) {
        std::cout << "Saved embedding: " << label
                  << " dim=" << emb.size()
                  << " -> " << embedDbPath << "\n";
    }
    return ok;
}

static void runOnFrame(const cv::Mat &frame,
                       const std::string &task,
                       const std::string &dbPath,
                       const std::string &embedDbPath,
                       p3::EmbeddingModel &embModel) {
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

    if (task == "--task4" || task == "--task5" || task == "--task6" || task == "--task9") {
        binary1 = p3::thresholdBinary(frame);
        cleaned = p3::morphCleanup(binary1, 3, 15);

        Task4Context ctx = computeMajorRegionFeatures(cleaned, 8, 500, 1);
        cv::Mat vis = frame.clone();

        std::vector<p3::DBSample> samples;
        p3::DBStats stats;
        bool dbOK = false;

        if (task == "--task6") {
            dbOK = p3::loadDB(dbPath, samples);
            if (dbOK) stats = p3::computeDBStats(samples);
        }

        std::vector<p3::EmbSample> embDB;
        bool embOK = false;

        if (task == "--task9") {
            embOK = p3::loadEmbeddingDB(embedDbPath, embDB);
        }

        for (size_t i = 0; i < ctx.features.size(); ++i) {
            p3::drawRegionOverlay(vis, ctx.features[i]);
            std::vector<double> fv = ctx.features[i].featureVector();

            if (task == "--task4" || task == "--task5") {
                std::cout << "Region " << ctx.features[i].regionId
                          << " feature vector: [";
                for (size_t k = 0; k < fv.size(); ++k) {
                    std::cout << fv[k] << (k + 1 < fv.size() ? ", " : "");
                }
                std::cout << "]\n";
            }

            if (task == "--task6" && dbOK) {
                double bestDist = 0.0;
                double thr = 20.0;
                std::string label = p3::classifyNNWithUnknown(fv, samples, stats, thr, bestDist);

                cv::Point org((int)ctx.features[i].centroid.x + 10,
                              (int)ctx.features[i].centroid.y + 25);

                cv::putText(vis, label, org,
                            cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            cv::Scalar(0, 0, 255), 2);

                std::cout << "Predicted: " << label
                          << " dist=" << bestDist << "\n";
            }

            if (task == "--task9") {
                std::vector<float> emb = embModel.computeEmbedding(frame, ctx.features[i]);

                std::string label = "UNKNOWN";
                double bestDist = 0.0;
                if (embOK && !emb.empty()) {
                    label = p3::classifyEmbeddingNN_SSD(emb, embDB, bestDist);
                }

                cv::Point org((int)ctx.features[i].centroid.x + 10,
                              (int)ctx.features[i].centroid.y + 60);

                cv::putText(vis, label, org,
                            cv::FONT_HERSHEY_SIMPLEX, 0.9,
                            cv::Scalar(0, 0, 255), 2);

                std::cout << "Embedding predicted: " << label
                          << " dist=" << bestDist
                          << " dim=" << emb.size() << "\n";
            }
        }

        cv::imshow("Input", frame);
        cv::imshow("Task2 Cleaned", cleaned);
        cv::imshow("Task4/6/9 Output", vis);
        return;
    }

    std::cerr << "Unknown task: " << task << "\n";
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
    std::string embedDbPath = "data/embed_db.txt";

    std::string onnxPath = "data/resnet18-v2-7.onnx";
    p3::EmbeddingModel embModel(onnxPath);

    if (task == "--task7") {
        if (mode != "--dir") {
            std::cerr << "Task7 requires --dir\n";
            return 1;
        }

        std::vector<std::string> classes =
            {"pen", "bolt", "ring", "scissors", "lock"};

        std::map<std::string, int> idx;
        for (int i = 0; i < (int)classes.size(); ++i)
            idx[classes[i]] = i;

        std::vector<std::vector<int>> confusion(
            classes.size(),
            std::vector<int>(classes.size(), 0));

        fs::path folder(arg);
        if (!fs::exists(folder) || !fs::is_directory(folder)) {
            std::cerr << "Not a directory: " << arg << "\n";
            return 1;
        }

        std::vector<p3::DBSample> samples;
        if (!p3::loadDB(dbPath, samples) || samples.empty()) {
            std::cerr << "DB load failed.\n";
            return 1;
        }
        p3::DBStats stats = p3::computeDBStats(samples);

        int total = 0;

        for (auto &e : fs::directory_iterator(folder)) {
            if (!e.is_regular_file() || !isImageFile(e.path()))
                continue;

            cv::Mat frame = cv::imread(e.path().string());
            if (frame.empty())
                continue;

            std::string trueLab =
                normalizeLabel(trueLabelFromFilename(e.path()));

            if (idx.find(trueLab) == idx.end())
                continue;

            cv::Mat binary1 = p3::thresholdBinary(frame);
            cv::Mat cleaned = p3::morphCleanup(binary1, 3, 15);
            Task4Context ctx = computeMajorRegionFeatures(cleaned, 8, 500, 1);

            if (ctx.features.empty())
                continue;

            std::vector<double> fv = ctx.features[0].featureVector();

            double bestDist = 0.0;
            double thr = 20.0;

            std::string pred =
                normalizeLabel(
                    p3::classifyNNWithUnknown(
                        fv, samples, stats, thr, bestDist));

            if (idx.find(pred) != idx.end()) {
                confusion[idx[trueLab]][idx[pred]]++;
            }

            std::cout << e.path().filename()
                      << " true=" << trueLab
                      << " pred=" << pred
                      << " dist=" << bestDist << "\n";

            total++;
        }

        std::cout << "\nConfusion Matrix (rows=true, cols=pred)\n\n";
        std::cout << std::setw(12) << " ";
        for (auto &c : classes)
            std::cout << std::setw(12) << c;
        std::cout << "\n";

        for (size_t r = 0; r < classes.size(); ++r) {
            std::cout << std::setw(12) << classes[r];
            for (size_t c = 0; c < classes.size(); ++c) {
                std::cout << std::setw(12)
                          << confusion[r][c];
            }
            std::cout << "\n";
        }

        std::cout << "\nTotal evaluated: "
                  << total << "\n";

        return 0;
    }

    if (mode == "--image") {
        cv::Mat frame = cv::imread(arg, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "Failed to read image\n";
            return 1;
        }

        runOnFrame(frame, task, dbPath, embedDbPath, embModel);

        while (true) {
            int key = cv::waitKey(0);
            if (key == 27 || key == 'q') break;

            if (task == "--task5" && key == 'N') {
                saveTrainingSampleFromLastFrame(frame, dbPath);
            }

            if (task == "--task9" && key == 'N') {
                saveEmbeddingSampleFromLastFrame(frame, embedDbPath, embModel);
            }
        }

        return 0;
    }

    if (mode == "--dir") {
        fs::path folder(arg);
        if (!fs::exists(folder) || !fs::is_directory(folder)) {
            std::cerr << "Not a directory\n";
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
            std::cerr << "No image files\n";
            return 1;
        }

        size_t idx = 0;
        while (true) {
            cv::Mat frame = cv::imread(files[idx].string(), cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "Failed to read: " << files[idx] << "\n";
                return 1;
            }

            runOnFrame(frame, task, dbPath, embedDbPath, embModel);

            int key = cv::waitKey(0);
            if (key == 27 || key == 'q') break;

            if (key == 'n' || key == ' ') {
                idx = (idx + 1) % files.size();
                continue;
            }

            if (task == "--task5" && key == 'N') {
                saveTrainingSampleFromLastFrame(frame, dbPath);
            }

            if (task == "--task9" && key == 'N') {
                saveEmbeddingSampleFromLastFrame(frame, embedDbPath, embModel);
            }
        }

        return 0;
    }

    std::cerr << "Unknown mode\n";
    showUsage();
    return 1;
}