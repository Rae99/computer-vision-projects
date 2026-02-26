#pragma once

#include "p3_segmentation.hpp"
#include <map>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace p3 {

// ── Embedding DB sample
// ───────────────────────────────────────────────────────
struct EmbSample {
    std::string label;
    std::vector<float> emb;
};

// ── ResNet18 embedding model
// ──────────────────────────────────────────────────
class EmbeddingModel {
  public:
    explicit EmbeddingModel(const std::string &onnxPath);

    std::vector<float> computeEmbedding(const cv::Mat &bgr,
                                        const RegionFeatures &region);

  private:
    cv::dnn::Net net;

    cv::Mat rotateAround(const cv::Mat &img, cv::Point2f center,
                         double angleDegrees) const;

    cv::Rect clippedRectFromPoints(const std::vector<cv::Point2f> &pts, int w,
                                   int h);

    cv::Mat extractAlignedROI(const cv::Mat &bgr,
                              const RegionFeatures &region) const;

    cv::Mat makeResNetBlob(const cv::Mat &roi224);
};

// ── DB helpers
// ────────────────────────────────────────────────────────────────
bool loadEmbeddingDB(const std::string &path, std::vector<EmbSample> &out);

bool appendEmbeddingSample(const std::string &path, const std::string &label,
                           const std::vector<float> &emb);

std::string classifyEmbeddingNN_SSD(const std::vector<float> &query,
                                    const std::vector<EmbSample> &db,
                                    double &bestDist);

} // namespace p3