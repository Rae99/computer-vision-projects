# Layer 3: CNN Architecture

**File:** `mnist_recognition.py` → `class MyNetwork`

---

## The Network at a Glance

```
Input: 28×28 grayscale image
  ↓
[conv1]   10 filters, 5×5  →  output: 10 channels, 24×24
[pool]    2×2 max pool + ReLU  →  10 channels, 12×12
  ↓
[conv2]   20 filters, 5×5  →  20 channels, 8×8
[dropout] 50% dropout
[pool]    2×2 max pool + ReLU  →  20 channels, 4×4
  ↓
[flatten] 20 × 4 × 4 = 320 numbers in a row
[fc1]     320 → 50 nodes + ReLU
[fc2]     50 → 10 nodes + log_softmax
  ↓
Output: 10 numbers (one per digit class), log-probabilities
```

---

## The Code

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 10 filters, 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 20 filters, 5x5
        self.dropout = nn.Dropout(p=0.5)               # 50% dropout
        self.pool = nn.MaxPool2d(2, 2)                 # 2x2 max pool
        self.fc1 = nn.Linear(320, 50)                  # 320 → 50
        self.fc2 = nn.Linear(50, 10)                   # 50 → 10

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))               # conv1 → pool → relu
        x = torch.relu(self.pool(self.dropout(self.conv2(x)))) # conv2 → dropout → pool → relu
        x = x.view(-1, 320)                                    # flatten
        x = torch.relu(self.fc1(x))                            # fc1 → relu
        x = torch.log_softmax(self.fc2(x), dim=1)             # fc2 → log_softmax
        return x
```

---

## Two Parts: `__init__` and `forward`

**`__init__`** — defines the *trainable components*. Think of this like declaring the state of a React component. Each layer has parameters (weights) that PyTorch will update during training.

**`forward`** — defines *how data flows* through those components. This is the function actually called when you do `model(images)`. Think of it like the `render()` method — it takes input and produces output.

---

## Why 320?

After two rounds of `conv(5×5) → pool(2×2)`:
- Start: 28×28
- After conv1 (5×5, no padding): 24×24
- After pool (2×2): 12×12
- After conv2 (5×5, no padding): 8×8
- After pool (2×2): 4×4
- With 20 filters: 20 × 4 × 4 = **320**

The `x.view(-1, 320)` call flattens the 3D feature maps into a 1D vector of 320 numbers.
(`-1` means "infer this dimension from batch size" — like `...rest` in JS spread)

---

## `nn.Module` — What Is It?

`MyNetwork` inherits from `nn.Module`, which is PyTorch's base class for all neural networks.

**What it gives you for free:**
- `model.parameters()` — iterator over all trainable weights (used by the optimizer)
- `model.train()` / `model.eval()` — toggle training vs. inference mode
- `model.state_dict()` — all weights as a dictionary (used for saving/loading)
- Automatic tracking of all sub-modules (conv1, fc1, etc.)

**Analogy:** Like extending a base `Component` class in React — you get lifecycle methods and state management for free, you just implement `forward` (like `render`).

---

## The `FlexibleNetwork` Variant (experiment.py)

`experiment.py` defines `FlexibleNetwork` — same architecture but with configurable parameters:

```python
class FlexibleNetwork(nn.Module):
    def __init__(self, conv1_filters=10, conv2_filters=20,
                 fc1_size=50, dropout_rate=0.5):
```

Used for the ablation study (Layer 8). The `flat_size = conv2_filters * 4 * 4` calculation is important — if you change the number of filters in conv2, the flattened size changes too.

---

## Deep Dives

- → `concepts/cnn-layers.md` (what conv, pool, dropout each do)
- → `concepts/forward-pass.md` (data shapes at each step)
- → `concepts/log-softmax-nllloss.md` (why log_softmax instead of softmax)
