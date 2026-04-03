## Project 5: Recognition using Deep Networks

```
Team: Junrui Ding, Junyao Han

Time travel days used: 0

------------------------------------------------------------
Dependencies
------------------------------------------------------------
Python 3.8+

pip install torch torchvision matplotlib

------------------------------------------------------------
Running the programs
------------------------------------------------------------

All scripts are in src/. Run from the project5/ directory.

Task 1 – Train CNN on MNIST
  python src/mnist_recognition.py
  MNIST data downloads automatically on first run (~11 MB).
  Saves trained model to mnist_model.pth.
  Outputs: first_six.png, training_curves.png

Task 1E-F – Test model on test set and custom digits
  python src/mnist_test.py
  Requires mnist_model.pth (run Task 1 first).
  Place custom handwritten digit images in my_digits/ to test them.

Task 2 – Visualize learned filters
  python src/mnist_examine.py
  Requires mnist_model.pth (run Task 1 first).
  Displays conv1 filter weights and their responses on a test image.

Task 3 – Transfer learning on Greek letters
  python src/greek_transfer.py
  Requires mnist_model.pth (run Task 1 first).
  Freezes pretrained layers, retrains final layer for alpha/beta/gamma.

Task 4 – MNIST with Vision Transformer
  python src/mnist_transformer.py
  Trains a ViT-style model on MNIST from scratch.

Task 5 – Ablation experiment
  python src/experiment.py
  Varies conv filter counts, FC size, and dropout; logs results to results.csv.

------------------------------------------------------------
Notes
------------------------------------------------------------
- data/ is git-ignored (auto-downloaded at runtime)
- output/ is git-ignored (generated files)
- my_digits/ is git-ignored (local test images)
```
