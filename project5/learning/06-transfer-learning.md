# Layer 6: Transfer Learning

**File:** `greek_transfer.py`

---

## The Core Idea

The MNIST network learned to recognize digits. Digits and Greek letters share low-level features: they're both pen-stroke symbols with curves, lines, and corners. Instead of training from scratch, we:

1. Load the pretrained MNIST weights
2. **Freeze** them — don't let them change during Greek training
3. **Replace** the last layer (10 outputs → 3 outputs for α, β, γ)
4. Train only the new last layer

**Analogy:** You have a React component library built for one product. A new product needs mostly the same components but with a different "export button." Instead of rebuilding the library, you import it as-is, freeze it (no modifications), and only swap out the export component.

---

## Loading and Freezing

```python
def build_greek_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))  # load pretrained MNIST weights

    # Freeze all parameters — gradients won't flow to these
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer: 50→10 becomes 50→3
    model.fc2 = nn.Linear(50, 3)
    # New layers have requires_grad=True by default
```

`requires_grad = False` is the key. It tells PyTorch: "don't compute gradients for this parameter during backprop." The optimizer only updates parameters with gradients, so frozen layers stay fixed.

Note: when you assign `model.fc2 = nn.Linear(50, 3)`, this new layer has fresh random weights AND `requires_grad=True`. Only the new layer will be trained.

---

## Why Only 27 Images?

The Greek dataset has only 9 images per class × 3 classes = 27 images total. Normally this is far too little to train a deep network (which has thousands of parameters). Transfer learning makes this feasible because:

- The frozen layers already know how to detect curves, lines, and strokes
- We're only training 50×3 + 3 = 153 parameters in the new `fc2` layer
- The pretrained features generalize well enough to the new task

---

## The Training Loop (same structure, different data)

```python
def train_greek(model, greek_loader, epochs=20):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

`model.parameters()` still iterates over all parameters, but frozen ones have `requires_grad=False` so the optimizer skips them automatically.

Higher `momentum=0.9` (vs. 0.5 for MNIST) because the dataset is tiny and we need the optimizer to keep moving confidently.

---

## ImageFolder — Loading from Directory Structure

```python
torchvision.datasets.ImageFolder(
    training_set_path,   # e.g., 'data/greek_train/'
    transform=...
)
```

`ImageFolder` assumes this directory structure:
```
data/greek_train/
├── alpha/   ← class 0
├── beta/    ← class 1
└── gamma/   ← class 2
```

It automatically assigns integer labels based on alphabetical folder order. No manual label file needed.

---

## The Transform Chain

```python
transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # PIL image → tensor [0, 1]
    GreekTransform(),                   # 133x133 color → 28x28 grayscale, inverted
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # same as MNIST
])
```

Order matters: `ToTensor()` must come first (PIL → tensor), then `GreekTransform()` works on the tensor, then normalize.

---

## Deep Dives

- → `concepts/transfer-learning.md` (why frozen features generalize)
