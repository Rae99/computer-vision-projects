# Concept: CNN Layers

## Plain English

A CNN (Convolutional Neural Network) is a stack of layers, each transforming the data in a specific way. Four types are used in this project:

---

## 1. Convolution Layer (`nn.Conv2d`)

**What it does:** Slides a small filter (e.g., 5×5) over the image. At each position, computes a weighted sum of the pixels under the filter. Produces one output value per position.

**Analogy:** Like Photoshop's "emboss" or "sharpen" filter. Except instead of being hand-designed, the values in the filter are *learned* during training.

**In code:**
```python
self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#                      ↑  ↑        ↑
#              in_channels  out_channels  filter_size
```
- `in_channels=1` — input is grayscale (1 channel)
- `out_channels=10` — 10 different filters, producing 10 "feature maps"
- Each filter detects one type of pattern: edge, curve, blob, etc.

**Shape effect:** Input 28×28 → output 24×24 (because a 5×5 filter centered at (0,0) needs pixels up to (4,4) — you lose 2 pixels on each side = 28-4=24)

---

## 2. Max Pooling (`nn.MaxPool2d`)

**What it does:** Divides the feature map into 2×2 blocks, keeps only the maximum value in each block. Halves the spatial dimensions.

**Analogy:** Like compressing a spreadsheet — take the max of every 2×2 group of cells, discard the rest.

**Why use it:**
- Reduces computation (smaller feature maps)
- Builds in translation invariance: a feature shifted by 1 pixel still activates the same output
- Keeps the strongest signal from each region

**Shape effect:** 24×24 → 12×12

---

## 3. ReLU Activation

**What it does:** `f(x) = max(0, x)` — zeroes out all negative values, passes positive values unchanged.

**Why it's needed:** Without nonlinearity, stacking linear layers is still just one linear operation. ReLU makes the network capable of learning complex, non-linear functions.

**Analogy:** Like a one-way valve — only lets positive signals through.

In `MyNetwork`, ReLU is applied after pooling, not as a separate layer:
```python
x = torch.relu(self.pool(self.conv1(x)))
```

---

## 4. Dropout (`nn.Dropout`)

**What it does:** During training, randomly sets 50% of neuron outputs to zero on each forward pass. Scales remaining values up to compensate.

**Why it helps:** Forces the network to not rely on any single neuron. Prevents overfitting (memorizing training data instead of generalizing).

**During inference:** `model.eval()` disables dropout — all neurons are active, multiplied by `(1 - dropout_rate)` to match expected scale.

**Analogy:** Like randomly muting half your team in a standup — forces everyone else to cover the gap, making the whole team more robust.

---

## 5. Fully Connected Layer (`nn.Linear`)

**What it does:** Every input neuron connects to every output neuron. The classic neural network layer.

**Analogy:** Like a matrix multiplication: `output = input × weights + bias`. In JS: `output = inputs.map(x => dot(x, weights) + bias)`.

In `MyNetwork`:
```python
self.fc1 = nn.Linear(320, 50)  # 320 inputs → 50 outputs
self.fc2 = nn.Linear(50, 10)   # 50 inputs → 10 class scores
```

---

## Code Before/After: CNN vs. Flat Network

**Without convolution** (naive):
```python
# Input: flatten 28×28 = 784 pixels directly to FC
self.fc1 = nn.Linear(784, 50)
self.fc2 = nn.Linear(50, 10)
```
Problem: doesn't capture spatial structure; 784 inputs = many parameters, slow to train.

**With convolution** (this project):
```python
# Conv layers extract spatial features, FC layers classify
self.conv1 = nn.Conv2d(1, 10, 5)
self.conv2 = nn.Conv2d(10, 20, 5)
self.fc1 = nn.Linear(320, 50)   # 320 << 784
self.fc2 = nn.Linear(50, 10)
```
Benefits: fewer parameters, translation invariance, spatial hierarchy.

---

## Where It Shows Up

| Layer | File | Line |
|-------|------|------|
| `conv1`, `conv2` | `mnist_recognition.py` | 19–20 |
| `pool`, `dropout`, `fc1`, `fc2` | `mnist_recognition.py` | 21–24 |
| All layers (flexible) | `experiment.py` | 23–34 |
| `conv1.weight` accessed | `mnist_examine.py` | 36 |
