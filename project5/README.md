## Project 5: Recognition using Deep Networks

```
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
  Outputs: first_six.png, training_curves.png

Task 1E-F – Test model on test set and custom digits
  python src/mnist_test.py
  Requires mnist_model.pth (run Task 1 first).
  Runs on both my_digits/ (original handwriting) and my_digits_v2/ (revised style).
  Outputs: output/custom_digits_v1.png, output/custom_digits_v2.png

Task 2 – Visualize learned filters
  python src/mnist_examine.py
  Requires mnist_model.pth (run Task 1 first).
  Displays conv1 filter weights and their responses on a test image.

Task 3 – Transfer learning on Greek letters
  python src/greek_transfer.py ../data/greek_train [../data/my_greek]
  Requires mnist_model.pth (run Task 1 first).
  Freezes pretrained layers, retrains final layer for alpha/beta/gamma.
  Optional second argument: path to custom handwritten Greek letter images.
  If greek_model.pth already exists, skips training and loads saved weights.

  Custom Greek letter examples (Task 3):
  https://drive.google.com/file/d/1FKZ-stSoXF-JJAYYxJFHOJTcUJQCz5WI/view?usp=drive_link

Task 4 – MNIST with Vision Transformer
  python src/mnist_transformer.py
  Trains a ViT-style model on MNIST from scratch.

Task 5 – Ablation experiment
  python src/experiment.py
  Varies conv filter counts, FC size, and dropout; logs results to output/experiment_results.csv.

------------------------------------------------------------
Notes
------------------------------------------------------------
- data/ is git-ignored (auto-downloaded at runtime)
- output/ is git-ignored (generated files)
- my_digits/ and my_greek/ are git-ignored (local test images)
```
