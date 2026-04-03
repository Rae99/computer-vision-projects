# Concept: Forward Pass & Data Shapes

## Plain English

The "forward pass" is what happens when you feed data into the network — it flows through each layer, changing shape and gaining meaning at each step, until it exits as class scores.

---

## Shape Trace Through MyNetwork

Starting with a batch of 64 images:

```
Input:          [64, 1, 28, 28]   # 64 images, 1 channel, 28×28

conv1(5×5):     [64, 10, 24, 24]  # 10 filters; 28-4=24 (no padding)
pool(2×2):      [64, 10, 12, 12]  # halved
relu:           [64, 10, 12, 12]  # shape unchanged, negatives zeroed

conv2(5×5):     [64, 20,  8,  8]  # 20 filters; 12-4=8
dropout:        [64, 20,  8,  8]  # shape unchanged, ~50% set to 0
pool(2×2):      [64, 20,  4,  4]  # halved
relu:           [64, 20,  4,  4]

flatten:        [64, 320]         # 20×4×4 = 320
fc1:            [64,  50]         # linear transform
relu:           [64,  50]

fc2:            [64,  10]         # 10 class scores (logits)
log_softmax:    [64,  10]         # 10 log-probabilities
```

---

## Reading Shape Notation

`[64, 1, 28, 28]` → `[batch_size, channels, height, width]`

In PyTorch, images are always `(N, C, H, W)` — batch first, then channel, then spatial dims. This is different from OpenCV/numpy which use `(H, W, C)`. That's why `.squeeze()` and `.unsqueeze()` are frequently used to add/remove dimensions.

---

## What `x.view(-1, 320)` Does

```python
x = x.view(-1, 320)  # reshape [64, 20, 4, 4] → [64, 320]
```

`view` is PyTorch's reshape. `-1` means "infer this dimension." Since `64 × 20 × 4 × 4 = 64 × 320`, PyTorch figures out the batch size is 64.

**Analogy:** Like `Array.flat()` in JS, applied to each image in the batch independently.

---

## model(images) vs. model.forward(images)

```python
# These are equivalent:
outputs = model(images)        # ← use this
outputs = model.forward(images) # ← works but not idiomatic
```

PyTorch's `nn.Module.__call__` does more than just call `forward` — it also runs registered hooks. Always call `model(x)`, not `model.forward(x)` directly.

---

## Where It Shows Up

| File | Call | Shape in/out |
|------|------|-------------|
| `mnist_recognition.py` | `outputs = model(images)` | [64,1,28,28] → [64,10] |
| `mnist_test.py` | `outputs = model(images)` | [10,1,28,28] → [10,10] |
| `mnist_test.py` | `output = model(tensor)` | [1,1,28,28] → [1,10] |
| `mnist_examine.py` | (filter application, not model call) | — |
