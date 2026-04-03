# Concept: Input Normalization

## The Magic Numbers: 0.1307 and 0.3081

These appear in every data-loading function:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

They're the **mean** and **standard deviation** of all pixel values in the MNIST training set.

This transform does: `pixel = (pixel - 0.1307) / 0.3081`

---

## Why Normalize?

Neural network weights are initialized around 0. If inputs are on a completely different scale (raw pixels: 0–1), gradients behave inconsistently and training is slow or unstable.

After normalization, inputs have approximately:
- Mean ≈ 0
- Standard deviation ≈ 1

This puts the input in the same range the network weights expect.

**Analogy:** Like standardizing units before doing statistics. If you're comparing heights and weights, you convert both to z-scores first — otherwise the weight in kg swamps the height in meters.

---

## The Tuple Syntax

```python
transforms.Normalize((0.1307,), (0.3081,))
#                    ↑ mean     ↑ std
#                    (one value per channel)
```

The tuples are because multi-channel images (RGB) have one mean/std per channel. MNIST is grayscale (1 channel), so there's one value each. The trailing comma makes it a tuple, not just parentheses.

For an RGB dataset like ImageNet: `Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))`.

---

## Greek Letter Images Use the Same Values

```python
# greek_transfer.py
torchvision.transforms.Normalize((0.1307,), (0.3081,))
```

Even though the Greek images are different content, we use MNIST's mean/std because the pretrained network was trained with those values. Using different normalization would shift the input distribution and hurt transfer learning performance.

---

## Where It Shows Up

| File | Context |
|------|---------|
| `mnist_recognition.py` | Training + test MNIST loader |
| `mnist_test.py` | Test loader + custom digit preprocessing |
| `mnist_examine.py` | Training image loader |
| `greek_transfer.py` | Greek dataset loader |
