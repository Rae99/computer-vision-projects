# CS5330 Project 3 – 2D Object Recognition System

Authors:
Junrui Ding,  Junyao Han  

---

## Overview

This project implements a complete 2D object recognition system using classical computer vision techniques together with a CNN-based embedding extension.

The system includes:

- Threshold-based foreground segmentation  
- Morphological filtering  
- Connected component analysis  
- Region-based feature extraction (implemented from scratch)  
- Feature database construction  
- Scaled Euclidean nearest-neighbor classification  
- Confusion matrix evaluation  
- One-shot classification using ResNet18 embeddings  

Baseline system uses two invariant geometric features:
- Percent filled  
- Aspect ratio  

The extension system uses 512-dimensional embeddings extracted from a pre-trained ResNet18 network.

---

## Project Structure

project3/
│
├── data/
│   ├── images/              # training images
│   ├── eval/                # evaluation images
│   ├── object_db.txt        # feature database
│   └── resnet18-v2-7.onnx   # CNN model (Task 9)
│
├── src/
│   ├── main.cpp
│   ├── p3_segmentation.cpp
│   ├── p3_db.cpp
│   ├── p3_embedding.cpp
│   └── ...
│
└── CMakeLists.txt

---

## Build

Using CLion or terminal:

cmake --build cmake-build-debug

Requirements:

- OpenCV installed
- data/resnet18-v2-7.onnx present for Task 9
- Working directory set to the project root

---

## Running the System (Batch Mode Only)

All commands below use directory mode.

---

### Task 1 – Thresholding

./main --task1 --dir data/images

Displays thresholded binary masks for all images in the folder.

---

### Task 2 – Morphological Filtering

./main --task2 --dir data/images

Displays cleaned binary masks after opening and closing operations.

---

### Task 3 – Connected Components Segmentation

./main --task3 --dir data/images

Shows segmented regions with small regions ignored and major regions highlighted.

---

### Task 4 – Region-Based Feature Extraction

./main --task4 --dir data/images

For each image, displays:

- Oriented bounding box  
- Axis of least central moment  
- Feature values printed in the console  

---

### Task 5 – Collect Training Data

./main --task5 --dir data/images

While browsing images:

- Press n or space to go to the next image  
- Press q or ESC to quit  
- Press N (uppercase) to save the current feature vector  

The system will prompt for a label and append the labeled feature vector to:

data/object_db.txt

---

### Task 6 – Baseline Classification

./main --task6 --dir data/eval

For each image:

- The predicted label is displayed on the image
- The predicted label and scaled Euclidean distance are printed to the console

---

### Task 7 – Performance Evaluation

./main --task7 --dir data/eval

Outputs:

- True label vs predicted label for each image  
- 5×5 confusion matrix  
- Total evaluated samples  
- Overall accuracy  

---

### Task 9 – One-Shot Embedding Classification (ResNet18)

Ensure the ONNX model exists:

data/resnet18-v2-7.onnx

Then run:

./main --task9 --dir data/eval

This performs classification using:

- 512-dimensional ResNet18 embeddings  
- SSD (sum-squared difference) distance  
- Full training or one-shot configuration depending on database setup  

---

## Results Summary

Baseline (2D handcrafted features):

- 100% accuracy on 20 evaluation images  
- Scaled Euclidean nearest neighbor  

Embedding (512-D ResNet18):

- 100% accuracy with full training set  
- 100% accuracy with one-shot (1 sample per class)  
- SSD distance metric  

---

## Demo Video (Task 8)

Demo link:

https://northeastern-my.sharepoint.com/:v:/g/personal/han_junya_northeastern_edu/IQBFP06xk6UgSokHWk715jm3AbtwRd0El5iBfT_og0Php4k

---

## Notes

- The system assumes uniform white background and consistent lighting.
- Objects must be fully contained within the image frame.
- Baseline features are efficient but sensitive to segmentation quality.
- CNN embeddings generalize better and support one-shot learning.
