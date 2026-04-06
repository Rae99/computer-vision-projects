# Project 5: Recognition using Deep Networks

Team: Junrui Ding, Junyao Han

Time travel days used: 0

------------------------------------------------------------
Dependencies
------------------------------------------------------------
Python 3.8+

Recommended: use a virtual environment so packages stay isolated.

  python3 -m venv venv
  source venv/bin/activate        # Windows: venv\Scripts\activate
  pip install torch torchvision matplotlib opencv-python

To activate the venv in future sessions:
  source venv/bin/activate

------------------------------------------------------------
Running the programs
------------------------------------------------------------

All scripts are in src/. Run from the project5/ directory.
Make sure the venv is active first (see above).

Task 1 – Train CNN on MNIST
  python src/mnist_recognition.py
  MNIST data downloads automatically on first run (~11 MB).
  Saves trained model to mnist_model.pth.
  Outputs: output/first_six.png, output/training_curves.png

Task 1E-F – Test model on test set and custom digits
  python src/mnist_test.py
  Requires mnist_model.pth (run Task 1 first).
  Runs on both my_digits/ (original handwriting) and my_digits_v2/ (revised style).
  Outputs: output/test_predictions.png, output/custom_digits_v1.png, output/custom_digits_v2.png

Task 2 – Visualize learned filters
  python src/mnist_examine.py
  Requires mnist_model.pth (run Task 1 first).
  Outputs: output/filters.png, output/filter_effects.png

Task 2 Extension – Analyze a pretrained ResNet18
  python src/ext2_pretrained_examine.py
  Loads a pretrained ResNet18 from torchvision and visualizes the first-layer filters.
  Output: output/ext2_resnet18_filters.png

Task 2 Extension – Replace conv1 with fixed Gabor filters
  python src/ext4_gabor_network.py
  Requires mnist_model.pth (run Task 1 first).
  Replaces conv1 with 10 fixed 5x5 Gabor filters, freezes conv1, and retrains the remaining layers.
  Outputs: output/ext4_gabor_filters.png, output/ext4_gabor_curves.png

Task 3 – Transfer learning on Greek letters
  python src/greek_transfer.py ../data/greek_train [../data/my_greek]
  Requires mnist_model.pth (run Task 1 first).
  Freezes pretrained layers and retrains the final layer for alpha/beta/gamma.
  Optional second argument: path to custom handwritten Greek letter images.
  If greek_model.pth already exists, skips training and loads saved weights.

  Custom Greek letter examples (Task 3):
  https://drive.google.com/file/d/1FKZ-stSoXF-JJAYYxJFHOJTcUJQCz5WI/view?usp=drive_link

Task 4 – MNIST with Vision Transformer
  python src/mnist_transformer.py
  Trains a ViT-style model on MNIST from scratch.
  Output: output/transformer_curves.png

Task 4 Extension – Transformer with CLS token
  python src/ext3_transformer_cls.py
  Trains a deeper transformer variant using a learnable CLS token instead of mean pooling.
  Outputs: output/ext3_transformer_cls_curves.png, ext3_transformer_cls.pth

Task 5 – Ablation experiment
  python src/experiment.py
  Runs a linear-search ablation study over 6 dimensions:
  conv1_filters, conv2_filters, dropout_rate, fc1_size, batch_size, and learning_rate.
  Outputs: output/experiment_results.csv, output/experiment_results.png

------------------------------------------------------------
Notes
------------------------------------------------------------
- data/ is git-ignored (auto-downloaded at runtime)
- output/ is git-ignored (generated files)
- my_digits/ and my_greek/ are git-ignored (local test images)
- If output/ does not exist, create it before running:
    mkdir -p output