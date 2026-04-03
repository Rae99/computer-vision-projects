# Concept: Transfer Learning

## Plain English

A network trained on task A has already learned useful representations. Transfer learning reuses those representations for task B, instead of training from scratch.

---

## What Gets Transferred?

Early layers of a CNN learn generic features: edges, textures, shapes. These are useful for almost any visual task — not just the original task.

In this project:
- conv1 and conv2 learned to detect strokes and curves from MNIST digits
- Those same features are useful for Greek letters (which are also stroke-based symbols)
- Only the final classification layer needs to change (10 classes → 3 classes)

**Analogy:** You learned to type on an English keyboard. Learning to type Portuguese doesn't require relearning finger placement — just a few new key combinations. The underlying skill transfers.

---

## The Three Steps

```python
# Step 1: Load pretrained weights
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))

# Step 2: Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace and unfreeze the head
model.fc2 = nn.Linear(50, 3)   # new layer: requires_grad=True by default
```

---

## Why Does Freezing Work?

`requires_grad = False` tells PyTorch: don't include this parameter in the computation graph for `.backward()`. The optimizer only receives gradients for non-frozen parameters.

During training:
- Gradient flows backward through fc2 (the new layer) → stops at frozen fc1
- Only fc2 weights are updated
- Pretrained conv1, conv2, fc1 stay exactly as trained on MNIST

---

## How Many Epochs Does It Take?

With only 27 images and only 153 trainable parameters, convergence is fast — often within 5–10 epochs. The code trains for 20 epochs to be safe.

Compare to MNIST: 60,000 images, ~20,000 parameters, 5 epochs.

---

## Code Before/After: With vs. Without Transfer

```python
# WITHOUT transfer learning: train from scratch on 27 images
model = MyNetwork()   # random weights
# → will overfit badly, likely < 50% accuracy

# WITH transfer learning: pretrained conv layers + new head
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
for param in model.parameters():
    param.requires_grad = False
model.fc2 = nn.Linear(50, 3)
# → often reaches 100% on 27 images within 20 epochs
```

---

## Where It Shows Up

| Step | File | Line(s) |
|------|------|---------|
| Load pretrained | `greek_transfer.py` | 49–50 |
| Freeze weights | `greek_transfer.py` | 53–54 |
| Replace head | `greek_transfer.py` | 57 |
| Train only head | `greek_transfer.py` | 66–93 |
