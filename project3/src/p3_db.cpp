#include "p3_db.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace p3 {

bool appendSample(const std::string &path, const std::string &label,
                  const std::vector<double> &fv, const std::string &imageName) {
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "appendSample: failed to open " << path << "\n";
        return false;
    }

    ofs << label;
    for (double v : fv) {
        ofs << " " << v;
    }
    if (!imageName.empty()) {
        ofs << " # " << imageName;
    }
    ofs << "\n";
    return true;
}

bool loadDB(const std::string &path, std::vector<DBSample> &outSamples) {
    outSamples.clear();

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "loadDB: failed to open " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty())
            continue;

        std::istringstream iss(line);
        DBSample s;
        if (!(iss >> s.label))
            continue;

        double v;
        while (iss >> v) {
            s.fv.push_back(v);
        }

        if (!s.label.empty() && !s.fv.empty()) {
            outSamples.push_back(s);
        }
    }

    if (outSamples.empty()) {
        std::cerr << "loadDB: db is empty or invalid: " << path << "\n";
        return false;
    }

    return true;
}

DBStats computeDBStats(const std::vector<DBSample> &samples) {
    DBStats st;
    if (samples.empty())
        return st;

    const int dim = (int)samples[0].fv.size();
    st.mean.assign(dim, 0.0);
    st.stdev.assign(dim, 0.0);

    // mean
    for (const auto &s : samples) {
        for (int i = 0; i < dim; ++i) {
            st.mean[i] += s.fv[i];
        }
    }
    for (int i = 0; i < dim; ++i) {
        st.mean[i] /= (double)samples.size();
    }

    // variance
    for (const auto &s : samples) {
        for (int i = 0; i < dim; ++i) {
            double d = s.fv[i] - st.mean[i];
            st.stdev[i] += d * d;
        }
    }

    // stdev (use population std; either is fine for scaling)
    for (int i = 0; i < dim; ++i) {
        st.stdev[i] = std::sqrt(st.stdev[i] / (double)samples.size());
        // avoid divide-by-zero
        if (st.stdev[i] < 1e-9)
            st.stdev[i] = 1.0;
    }

    return st;
}

double scaledEuclidean(const std::vector<double> &a,
                       const std::vector<double> &b,
                       const std::vector<double> &stdev) {
    const int dim = (int)a.size();
    double sum = 0.0;

    for (int i = 0; i < dim; ++i) {
        double s = stdev[i];
        if (s < 1e-9)
            s = 1.0;
        double z = (a[i] - b[i]) / s;
        sum += z * z;
    }
    return sum; // squared distance is fine for nearest neighbor
}

std::string classifyNN(const std::vector<double> &query,
                       const std::vector<DBSample> &samples,
                       const DBStats &stats, double &bestDist) {
    bestDist = 1e100;
    std::string bestLabel = "UNKNOWN";

    if (samples.empty())
        return bestLabel;

    for (const auto &s : samples) {
        if (s.fv.size() != query.size())
            continue;
        double d = scaledEuclidean(query, s.fv, stats.stdev);
        if (d < bestDist) {
            bestDist = d;
            bestLabel = s.label;
        }
    }
    return bestLabel;
}

std::string classifyNNWithUnknown(const std::vector<double> &query,
                                  const std::vector<DBSample> &samples,
                                  const DBStats &stats, double threshold,
                                  double &bestDist) {
    std::string label = classifyNN(query, samples, stats, bestDist);
    if (bestDist > threshold)
        return "UNKNOWN";
    return label;
}

} // namespace p3