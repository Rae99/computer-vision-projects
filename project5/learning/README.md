# Project 5 Learning Guide
## Recognition using Deep Networks

> You're a CS student with React/Node experience reading code written by Claude.
> This guide walks you through the code layer by layer, most familiar → most novel.
> Each layer has its own file. Concepts that show up across files get their own deep-dive in `concepts/`.

---

## Layered Roadmap

| # | Layer | File | Key Question | Status |
|---|-------|------|-------------|--------|
| 1 | Project structure & entry points | `01-project-structure.md` | How is the code organized? Where does it start? | ⬜ |
| 2 | Data loading & preprocessing | `02-data-loading.md` | How does image data get into the model? | ⬜ |
| 3 | CNN architecture | `03-cnn-architecture.md` | What does `MyNetwork` actually build? | ⬜ |
| 4 | Training loop | `04-training-loop.md` | How does the network learn? | ⬜ |
| 5 | Filter visualization | `05-filter-visualization.md` | What did the network actually learn to see? | ⬜ |
| 6 | Transfer learning | `06-transfer-learning.md` | How do you reuse a trained network for a new task? | ⬜ |
| 7 | Vision Transformer | `07-vision-transformer.md` | What's the alternative to convolutions? | ⬜ |
| 8 | Ablation experiment | `08-ablation-experiment.md` | How do you systematically test architectural choices? | ⬜ |

**How to use:** Work through layers in order. When you finish a layer, mark it ✅ above.

---

## File Map

```
src/
├── mnist_recognition.py   → Layers 2, 3, 4  (core CNN + training)
├── mnist_test.py          → Layer 2          (loading + custom digit preprocessing)
├── mnist_examine.py       → Layer 5          (filter visualization)
├── greek_transfer.py      → Layer 6          (transfer learning)
├── mnist_transformer.py   → Layer 7          (Vision Transformer)
└── experiment.py          → Layer 8          (ablation study)
```

---

## Concepts Index

| Concept | File | One-liner |
|---------|------|-----------|
| CNN layers (conv, pool, dropout, FC) | `concepts/cnn-layers.md` | The building blocks and what each one does |
| Forward pass | `concepts/forward-pass.md` | How data flows through the network |
| Training loop (loss, backprop, optimizer) | `concepts/training-loop.md` | How the network learns from mistakes |
| log_softmax + NLLLoss | `concepts/log-softmax-nllloss.md` | Why these two are always used together |
| Normalization (mean/std) | `concepts/normalization.md` | Why we subtract 0.1307 and divide by 0.3081 |
| Transfer learning | `concepts/transfer-learning.md` | Reusing a pretrained network for a new task |
| Vision Transformer (ViT) | `concepts/vision-transformer.md` | Treating image patches like words in a sentence |

---

## Vocabulary Glossary

| Term | Plain English |
|------|--------------|
| **Tensor** | A multi-dimensional array — like a JS nested array, but GPU-friendly and type-strict |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | A small chunk of the dataset processed together (like pagination in an API) |
| **Loss** | A number measuring how wrong the model's predictions are — lower is better |
| **Gradient** | The direction and amount each weight should change to reduce loss |
| **Backpropagation** | Automatically computing gradients through the whole network (chain rule) |
| **Optimizer** | The algorithm that actually updates the weights using gradients (SGD, Adam) |
| **Convolution** | A filter sliding over an image, detecting patterns like edges or curves |
| **Max pooling** | Downsampling: keep only the strongest signal in each region |
| **Dropout** | Randomly zero out some neurons during training to prevent memorization |
| **ReLU** | Activation function: `max(0, x)` — kills negative values, passes positives through |
| **Softmax** | Converts raw scores to probabilities that sum to 1 |
| **log_softmax** | log(softmax(x)) — numerically more stable, used with NLLLoss |
| **Transfer learning** | Taking a network trained on task A and adapting it for task B |
| **Fine-tuning** | Training only some layers of a pretrained model |
| **Token** | In transformers: a fixed-size vector representing a chunk of input |
| **Attention** | A mechanism letting each part of the input "look at" and weigh other parts |
| **Patch** | A small rectangular crop of an image, used as input to a transformer |
| **Embedding** | A learned vector representation of something (a patch, a word, etc.) |
| **Ablation study** | Testing variations of a design to measure the effect of each component |
