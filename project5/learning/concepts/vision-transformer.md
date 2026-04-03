# Concept: Vision Transformer (ViT)

## Plain English

A Vision Transformer treats an image like a sentence. It chops the image into small patches, converts each patch to a vector (like a word embedding), and then uses a transformer to let patches "talk to each other" and figure out what the image shows.

---

## The Key Ingredient: Self-Attention

Self-attention lets each token look at every other token and decide how much to "attend" to each.

For a sentence "The cat sat on the mat":
- "sat" attends strongly to "cat" (who is doing the sitting) and "mat" (where)
- This context is how transformers understand meaning

For a digit image split into 16 patches:
- The bottom-right patch of a "7" might attend strongly to the top-horizontal-bar patch
- Attention captures these long-range spatial relationships

**Analogy:** CNN is like reading a book word by word, only looking at neighbors. Transformer is like reading the whole book and highlighting which passages relate to each other.

---

## Why Patches?

Transformers process sequences. A 28×28 image has 784 pixels — attending over 784 tokens is expensive (attention is O(n²)). By using 7×7 patches, we reduce to 16 tokens. Attention over 16 tokens is cheap.

---

## Multi-Head Attention

```python
nhead=4  # 4 attention heads
```

Instead of one attention computation, run 4 in parallel, each learning different relationship types:
- Head 1 might learn horizontal relationships (left patch attends to right patch)
- Head 2 might learn vertical relationships
- Head 3 might learn local vs. global structure
- Head 4 might learn something else entirely

The outputs of all 4 heads are concatenated and projected back to `d_model` dimensions.

---

## Positional Embedding: Why Needed

Attention is permutation-invariant — it doesn't care about order. If you shuffle the 16 patches, the same attention weights apply. That's wrong — position matters in images.

Positional embeddings add a learned bias to each patch's token based on its position. After training, the network learns: "patch in position 0 (top-left) should be interpreted differently from patch in position 15 (bottom-right)."

```python
self.pos_embedding = nn.Parameter(torch.randn(1, 16, 64))
x = x + self.pos_embedding  # add position info to each token
```

---

## The CLS Token Alternative (not used here)

BERT and the original ViT add a special "classification token" (CLS) to the beginning of the sequence:
```
[CLS, patch_0, patch_1, ..., patch_15]
```
After the transformer, only the CLS token is used for classification — it's been trained to aggregate information from all other tokens.

This project uses **average pooling** instead (simpler, works fine for small images):
```python
x = x.mean(dim=1)  # average all 16 patch tokens
```

---

## Code Before/After: CNN vs. Transformer Forward Pass

```python
# CNN (MyNetwork.forward)
x = relu(pool(conv1(x)))          # local filters
x = relu(pool(dropout(conv2(x)))) # local filters
x = x.view(-1, 320)               # flatten
x = relu(fc1(x))

# Transformer (NetTransformer.forward)
x = unfold_into_patches(x)        # 16 patches × 49 pixels
x = patch_embedding(x)            # 16 patches × 64 dims
x = x + pos_embedding             # add position info
x = transformer(x)                # patches attend to each other
x = x.mean(dim=1)                 # aggregate
x = classifier(x)
```

---

## Where It Shows Up

| Component | File | Lines |
|-----------|------|-------|
| Full ViT implementation | `mnist_transformer.py` | 16–75 |
| Patch splitting (`unfold`) | `mnist_transformer.py` | 57–58 |
| Patch + positional embedding | `mnist_transformer.py` | 29–32, 64 |
| TransformerEncoder | `mnist_transformer.py` | 35–42 |
| Average pooling + head | `mnist_transformer.py` | 70–73 |
