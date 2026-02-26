#include "p3_embedding.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace p3 {

EmbeddingModel::EmbeddingModel(const std::string &onnxPath) {
    net = cv::dnn::readNetFromONNX(onnxPath);
    if (net.empty()) {
        std::cerr << "Failed to load ONNX model: " << onnxPath << "\n";
    }
}

cv::Mat EmbeddingModel::rotateAround(const cv::Mat &img, cv::Point2f center,
                                     double angleDegrees) const {
    cv::Mat rot = cv::getRotationMatrix2D(center, angleDegrees, 1.0);
    cv::Mat out;
    cv::warpAffine(img, out, rot, img.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);
    return out;
}

cv::Rect
EmbeddingModel::clippedRectFromPoints(const std::vector<cv::Point2f> &pts,
                                      int w, int h) {
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (const auto &p : pts) {
        minx = std::min(minx, p.x);
        miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x);
        maxy = std::max(maxy, p.y);
    }

    int x = (int)std::floor(minx);
    int y = (int)std::floor(miny);
    int rw = (int)std::ceil(maxx) - x;
    int rh = (int)std::ceil(maxy) - y;

    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    rw = std::max(1, std::min(rw, w - x));
    rh = std::max(1, std::min(rh, h - y));

    return cv::Rect(x, y, rw, rh);
}

cv::Mat EmbeddingModel::extractAlignedROI(const cv::Mat &bgr,
                                          const RegionFeatures &region) const {
    CV_Assert(!bgr.empty());

    double angleDeg = -region.thetaMajor * 180.0 / CV_PI;
    cv::Mat rotated = rotateAround(bgr, region.centroid, angleDeg);

    cv::Mat rotM = cv::getRotationMatrix2D(region.centroid, angleDeg, 1.0);

    std::vector<cv::Point2f> rotCorners;
    rotCorners.reserve(4);
    for (int i = 0; i < 4; ++i) {
        cv::Point2f p = region.obbCorners[i];
        double x = rotM.at<double>(0, 0) * p.x + rotM.at<double>(0, 1) * p.y +
                   rotM.at<double>(0, 2);
        double y = rotM.at<double>(1, 0) * p.x + rotM.at<double>(1, 1) * p.y +
                   rotM.at<double>(1, 2);
        rotCorners.push_back(cv::Point2f((float)x, (float)y));
    }

    cv::Rect roiRect =
        const_cast<EmbeddingModel *>(this)->clippedRectFromPoints(
            rotCorners, rotated.cols, rotated.rows);

    return rotated(roiRect).clone();
}

cv::Mat EmbeddingModel::makeResNetBlob(const cv::Mat &roi224) {
    CV_Assert(!roi224.empty());

    cv::Mat blob =
        cv::dnn::blobFromImage(roi224, 1.0 / 255.0, cv::Size(224, 224),
                               cv::Scalar(0, 0, 0), true, false);

    // blob shape: [1, 3, 224, 224]
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    int H = blob.size[2];
    int W = blob.size[3];

    for (int c = 0; c < 3; ++c) {
        // pointer to channel c plane
        float *ptr = blob.ptr<float>(0, c);
        for (int i = 0; i < H * W; ++i) {
            ptr[i] = (ptr[i] - mean[c]) / stdv[c];
        }
    }

    return blob;
}

std::vector<float>
EmbeddingModel::computeEmbedding(const cv::Mat &bgr,
                                 const RegionFeatures &region) {
    std::vector<float> emb;
    if (net.empty())
        return emb;

    cv::Mat roi = extractAlignedROI(bgr, region);
    if (roi.empty())
        return emb;

    cv::Mat roi224;
    cv::resize(roi, roi224, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    cv::Mat blob = makeResNetBlob(roi224);
    net.setInput(blob);
    cv::Mat out = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    out = out.reshape(1, 1);
    emb.resize((size_t)out.cols);
    for (int i = 0; i < out.cols; ++i)
        emb[(size_t)i] = out.at<float>(0, i);

    return emb;
}

// ── DB helpers
// ────────────────────────────────────────────────────────────────

bool loadEmbeddingDB(const std::string &path, std::vector<EmbSample> &out) {
    out.clear();
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Could not open embedding DB: " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        EmbSample s;
        if (!(iss >> s.label))
            continue;
        float v;
        while (iss >> v)
            s.emb.push_back(v);
        if (!s.label.empty() && !s.emb.empty())
            out.push_back(std::move(s));
    }
    return !out.empty();
}

bool appendEmbeddingSample(const std::string &path, const std::string &label,
                           const std::vector<float> &emb) {
    if (label.empty() || emb.empty())
        return false;
    std::ofstream fout(path, std::ios::app);
    if (!fout.is_open()) {
        std::cerr << "Could not open embedding DB for append: " << path << "\n";
        return false;
    }
    fout << label;
    for (float v : emb)
        fout << " " << v;
    fout << "\n";
    return true;
}

static double ssd(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty())
        return 1e18;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum;
}

std::string classifyEmbeddingNN_SSD(const std::vector<float> &query,
                                    const std::vector<EmbSample> &db,
                                    double &bestDist) {
    bestDist = 1e18;
    std::string best = "UNKNOWN";
    if (query.empty() || db.empty())
        return best;
    for (const auto &s : db) {
        double d = ssd(query, s.emb);
        if (d < bestDist) {
            bestDist = d;
            best = s.label;
        }
    }
    return best;
}

} // namespace p3
