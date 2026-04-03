# Layer 5: Filter Visualization

**File:** `mnist_examine.py`

---

## What This File Does

After training, the network has learned weights. This file makes those weights visible — you can literally see what patterns each convolution filter is looking for.

Two visualizations:
1. **The filters themselves** — what 5×5 pattern each of the 10 conv1 filters detects
2. **Filter responses** — what each filter "sees" when applied to a real image

---

## Accessing Weights

```python
weights = model.conv1.weight  # shape: [10, 1, 5, 5]
```

- `10` — 10 filters (one per row in the plot)
- `1` — 1 input channel (grayscale)
- `5, 5` — each filter is 5×5 pixels

To get filter `i`:
```python
filt = weights[i, 0]  # shape: [5, 5]
```

**Analogy:** Like reading the weights of a trained model the same way you'd read any object attribute. `model.conv1` is a `nn.Conv2d` layer object; its `.weight` attribute is a tensor of all the filter values.

---

## `torch.no_grad()` Context

```python
with torch.no_grad():
    for i in range(10):
        filt = weights[i, 0].numpy()
```

When you access `.numpy()` on a tensor, PyTorch needs it to not be tracked in the computation graph. `torch.no_grad()` disables gradient tracking. You'd also use this any time you're just reading/displaying weights, not training.

---

## Visualizing Filters with matplotlib

```python
fig, axes = plt.subplots(3, 4, figsize=(10, 7))  # 3 rows × 4 cols = 12 slots for 10 filters
with torch.no_grad():
    for i in range(10):
        row, col = divmod(i, 4)  # divmod(7, 4) = (1, 3) → row 1, col 3
        filt = weights[i, 0].numpy()
        axes[row, col].imshow(filt, cmap='viridis')
```

`divmod(i, 4)` maps a flat index `i` to `(row, col)` in a 4-column grid — same math as `Math.floor(i/4)` and `i%4` in JavaScript.

The last 2 slots (indices 10 and 11) are hidden with `axes[row, col].axis('off')`.

---

## Applying Filters with OpenCV

```python
filtered = cv2.filter2D(img_np, -1, filt)
```

`filter2D` slides the 5×5 filter over the image and computes a weighted sum at each position — this is exactly what `nn.Conv2d` does internally during the forward pass.

- `img_np` — the image as a numpy array (converted from tensor with `.squeeze().numpy()`)
- `-1` — output depth same as input depth
- `filt` — the 5×5 weight matrix

**Why manually apply it here instead of running through the model?** Because we want to isolate just the effect of filter `i`, not the full network. The model's forward pass applies all 10 filters simultaneously and then adds pooling, ReLU, etc.

---

## What Do the Filter Responses Tell You?

The filters in conv1 learn to detect low-level features: edges (horizontal, vertical, diagonal), blobs of light/dark, etc. When you apply filter `i` to an image:
- **Bright areas** in the response = places where that filter's pattern strongly matches
- **Dark areas** = places where it doesn't match

This is how CNNs build up recognition: conv1 finds edges → conv2 combines edges into shapes → fully connected layers vote on the final class.

---

## Deep Dives

- → `concepts/cnn-layers.md` (what convolution actually computes)
