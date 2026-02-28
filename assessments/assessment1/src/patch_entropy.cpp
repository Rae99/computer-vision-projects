// compile: g++ patch_entropy.cpp -o patch_entropy `pkg-config --cflags --libs
// opencv4`

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>

double computeEntropy(const cv::Mat &grayImg) {
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = range;
    cv::Mat hist;
    cv::calcHist(&grayImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    hist /= (float)grayImg.total();

    double entropy = 0.0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 0)
            entropy -= p * log2(p);
    }
    return entropy;
}

cv::Mat drawHistogram(const cv::Mat &grayPatch, int width = 256,
                      int height = 150) {
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = range;
    cv::Mat hist;
    cv::calcHist(&grayPatch, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    cv::normalize(hist, hist, 0, height, cv::NORM_MINMAX);

    cv::Mat histImg(height, width, CV_8UC1, cv::Scalar(0)); // 黑底
    for (int i = 0; i < histSize; i++) {
        int val = (int)hist.at<float>(i);
        cv::line(histImg, cv::Point(i, height), cv::Point(i, height - val),
                 cv::Scalar(255), 1); // 白线
    }
    return histImg;
}

int main() {
    cv::Mat img = cv::imread("../images/photo.jpg");
    if (img.empty()) {
        std::cerr << "Could not load image!" << std::endl;
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Define patches manually — adjust (x, y, width, height) to match your
    // photo Patch 1: sky region (uniform, expected low entropy) upper middle
    cv::Rect skyRect(1000, 20, 3000, 880);
    // Patch 2: grass/texture region (middle entropy) lower half
    cv::Rect grassRect(0, 2350, 6000, 1600);
    // Patch 3: tree and sky region (high entropy) middle where tree meets
    // sky
    cv::Rect treeRect(1700, 1370, 4300, 900);

    cv::Mat skyPatch = gray(skyRect);
    cv::Mat grassPatch = gray(grassRect);
    cv::Mat treePatch = gray(treeRect);

    std::cout << "Sky patch   entropy: " << computeEntropy(skyPatch)
              << std::endl;
    std::cout << "Grass patch entropy: " << computeEntropy(grassPatch)
              << std::endl;
    std::cout << "Tree patch  entropy: " << computeEntropy(treePatch)
              << std::endl;

    // Visualize: draw rectangles on original image
    cv::rectangle(img, skyRect, cv::Scalar(255, 0, 0), 8);   // blue = sky
    cv::rectangle(img, grassRect, cv::Scalar(0, 255, 0), 8); // green = grass
    cv::rectangle(img, treeRect, cv::Scalar(0, 0, 255), 8);  // red = tree

    cv::putText(img, "Sky: 4.03", cv::Point(skyRect.x, skyRect.y + 250),
                cv::FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(255, 0, 0), 8);

    cv::putText(img, "Trees: 7.28", cv::Point(treeRect.x, treeRect.y + 250),
                cv::FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 0, 255), 8);

    cv::putText(img, "Grass: 6.61", cv::Point(grassRect.x, grassRect.y + 250),
                cv::FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(0, 255, 0), 8);

    cv::imwrite("../images/patches_annotated.jpg", img);

    // Save individual patches
    cv::imwrite("../images/patch_sky.jpg", skyPatch);
    cv::imwrite("../images/patch_grass.jpg", grassPatch);
    cv::imwrite("../images/patch_tree.jpg", treePatch);

    // Save histograms for each patch
    cv::Mat histSky = drawHistogram(skyPatch);
    cv::Mat histGrass = drawHistogram(grassPatch);
    cv::Mat histTree = drawHistogram(treePatch);

    cv::imwrite("../images/hist_sky.png", histSky);
    cv::imwrite("../images/hist_grass.png", histGrass);
    cv::imwrite("../images/hist_tree.png", histTree);

    return 0;
}