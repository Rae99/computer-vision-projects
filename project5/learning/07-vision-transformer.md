# Layer 7: Vision Transformer (ViT)

**File:** `mnist_transformer.py` → `class NetTransformer`

---

## The Big Idea: Images as Sequences

CNNs think of images spatially — filters slide over neighboring pixels.
Transformers think of images as **sequences of patches** — like words in a sentence.

The 28×28 MNIST image is divided into 4×4 = **16 patches**, each 7×7 pixels. Each patch is converted to a vector ("token"), and the transformer learns relationships between all patches simultaneously.

**Analogy:** Instead of reading a book character by character (CNN: pixel by pixel), you read it by paragraph (ViT: patch by patch), and you can see how paragraph 1 relates to paragraph 16 directly.

---

## Architecture at a Glance

```
Input: 28×28 image
  ↓
[Patch splitting]     16 patches of 7×7 = 49 pixels each
[Patch embedding]     Linear: 49 → 64 dimensions per patch  (sequence: 16 × 64)
[+ Positional embed]  Add learned position info to each token
  ↓
[Transformer encoder] 2 layers of multi-head self-attention
  ↓
[Average pooling]     Average all 16 tokens → single 64-dim vector
[Classifier]          Linear(64→128) → ReLU → Linear(128→10) → log_softmax
  ↓
Output: 10 log-probabilities
```

---

## Patch Splitting (the unfold trick)

```python
def forward(self, x):
    B, C, H, W = x.shape          # e.g., (64, 1, 28, 28)
    p = self.patch_size            # 7

    x = x.unfold(2, p, p).unfold(3, p, p)   # B × 1 × 4 × 4 × 7 × 7
    x = x.contiguous().view(B, -1, p * p)   # B × 16 × 49
```

`unfold(dim, size, step)` slides a window of `size` with step `step` along dimension `dim`. With `size=step=7` on a 28-pixel dimension, you get exactly 4 non-overlapping windows.

The final `view(B, -1, p*p)` reshapes into a sequence: each of the 16 patches is now a flat vector of 49 numbers.

---

## Patch Embedding

```python
self.patch_embedding = nn.Linear(self.patch_dim, d_model)  # 49 → 64
```

A single linear layer maps each 49-pixel patch to a 64-dimensional vector. This is learned — the network figures out which combinations of pixels in a patch are meaningful.

**Analogy:** Like a word embedding in NLP — maps a raw word (one-hot vector) to a dense semantic vector.

---

## Positional Embedding

```python
self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
# shape: (1, 16, 64)
```

Transformers have no inherent notion of order — attention is a set operation. Positional embeddings add location information: "this token came from the top-left patch."

`nn.Parameter` means this tensor is trainable — it's learned alongside the network weights, not hard-coded.

```python
x = x + self.pos_embedding  # broadcast across batch: (B, 16, 64) + (1, 16, 64)
```

---

## Transformer Encoder

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64,          # token dimension
    nhead=4,             # 4 attention heads
    dim_feedforward=128, # hidden size in the MLP sublayer
    dropout=0.1,
    batch_first=True     # input shape is (batch, seq, features) — not (seq, batch, features)
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

Each `TransformerEncoderLayer` does:
1. **Multi-head self-attention** — each patch looks at all other patches and weighs their importance
2. **Feed-forward MLP** — processes each token independently (like a pointwise conv)

With 4 heads, each head learns a different type of relationship between patches.

---

## Average Pooling to Get One Vector

```python
x = x.mean(dim=1)  # (B, 16, 64) → (B, 64)
```

Averages all 16 patch tokens into a single 64-dim vector representing the whole image. Alternative: use a special CLS token (like BERT) — but averaging is simpler and works fine here.

---

## Classifier Head

```python
self.classifier = nn.Sequential(
    nn.Linear(d_model, mlp_dim),   # 64 → 128
    nn.ReLU(),
    nn.Linear(mlp_dim, num_classes) # 128 → 10
)
```

Standard MLP classifier on top of the transformer output. Same idea as `fc1` + `fc2` in the CNN.

---

## CNN vs. Transformer

| | CNN (`MyNetwork`) | Transformer (`NetTransformer`) |
|--|---|---|
| Inductive bias | Locality (nearby pixels matter more) | None — any patch can attend to any patch |
| Good for | Small datasets, clear spatial structure | Large datasets, long-range dependencies |
| Optimizer | SGD | Adam (transformers need adaptive LR) |
| Interpretable? | Yes — filter visualization | Harder — attention maps help |

---

## Deep Dives

- → `concepts/vision-transformer.md` (self-attention explained from scratch)
