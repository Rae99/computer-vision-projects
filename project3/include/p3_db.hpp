#ifndef P3_DB_HPP
#define P3_DB_HPP

#include <string>
#include <vector>

namespace p3 {

    struct DBSample {
        std::string label;
        std::vector<double> fv;  // feature vector
    };

    struct DBStats {
        std::vector<double> mean;
        std::vector<double> stdev;
    };

    // Task5: append a labeled feature vector to db file
    bool appendSample(const std::string &path,
                      const std::string &label,
                      const std::vector<double> &fv);

    // Load all samples from db file
    bool loadDB(const std::string &path, std::vector<DBSample> &outSamples);

    // Compute per-dimension mean/std for scaling
    DBStats computeDBStats(const std::vector<DBSample> &samples);

    // Scaled Euclidean distance: sum_i ((a_i - b_i)/stdev_i)^2
    double scaledEuclidean(const std::vector<double> &a,
                           const std::vector<double> &b,
                           const std::vector<double> &stdev);

    // Nearest-neighbor classification
    // Returns predicted label, and outputs bestDist (scaled distance)
    std::string classifyNN(const std::vector<double> &query,
                           const std::vector<DBSample> &samples,
                           const DBStats &stats,
                           double &bestDist);

    // Optional extension: unknown detection by threshold
    // If bestDist > threshold -> return "UNKNOWN"
    std::string classifyNNWithUnknown(const std::vector<double> &query,
                                      const std::vector<DBSample> &samples,
                                      const DBStats &stats,
                                      double threshold,
                                      double &bestDist);

} // namespace p3

#endif