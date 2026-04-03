# Layer 8: Ablation Experiment

**File:** `experiment.py`

---

## What Is an Ablation Study?

An ablation study systematically removes or changes one component of a model at a time, measuring the effect on performance. The name comes from surgery — "ablation" means removing tissue to study its function.

**Analogy:** Like A/B testing your architecture. Hold everything else constant, change one thing, see what the metrics say.

---

## The FlexibleNetwork

```python
class FlexibleNetwork(nn.Module):
    def __init__(self, conv1_filters=10, conv2_filters=20,
                 fc1_size=50, dropout_rate=0.5):
```

Same architecture as `MyNetwork`, but 4 parameters are configurable. The key math:

```python
flat_size = conv2_filters * 4 * 4
self.fc1 = nn.Linear(flat_size, fc1_size)
```

If `conv2_filters` changes, the flattened size changes too — `fc1` must be updated accordingly. This is the main reason a separate `FlexibleNetwork` class was needed instead of just modifying `MyNetwork`.

---

## The Search Strategy: Linear Search (Round-Robin)

Rather than testing every combination of all parameters (which would be L×M×N experiments), the code uses a **linear search**:

1. Hold all parameters at baseline, vary `conv1_filters` → find best value
2. Use best `conv1_filters`, vary `dropout_rate` → find best value
3. Use best `conv1_filters` + `dropout_rate`, vary `fc1_size`

```python
baseline = {'conv1': 10, 'conv2': 20, 'fc1': 50, 'dropout': 0.5}

# Dimension 1: conv1_filters = [5, 10, 20, 40]
for v in conv1_options:
    model = FlexibleNetwork(conv1_filters=v, ...)
    acc = quick_train(...)
    if acc > best_acc:
        best_conv1 = v

# Dimension 2: dropout = [0.1, 0.25, 0.5, 0.75]  ← uses best_conv1
for v in dropout_options:
    model = FlexibleNetwork(conv1_filters=best_conv1, dropout_rate=v, ...)
```

4 + 4 + 4 = **12 experiments** instead of 4×4×4 = 64.

---

## quick_train()

```python
def quick_train(model, train_loader, test_loader, epochs=3):
    # ... trains for 3 epochs ...
    return correct / total  # final test accuracy
```

Only 3 epochs per variant (vs. 5 for the full model) to keep experiments fast. The assumption is that 3-epoch relative performance is a good proxy for full-training performance.

---

## The Hypotheses (in the code comments)

```python
# 1. More conv filters → higher accuracy, up to a point (diminishing returns)
# 2. Higher dropout → lower accuracy (too much regularization with limited data)
# 3. Larger FC1 → marginal improvement (bottleneck is in conv layers for MNIST)
```

These are the predictions made *before* running experiments — good scientific practice.

---

## Results Storage

```python
results.append({'dimension': 'conv1_filters', 'value': v, 'accuracy': acc})
# ...
# Saved to CSV:
writer = csv.DictWriter(f, fieldnames=['dimension', 'value', 'accuracy'])
```

Each result is a dict with three keys. The CSV format makes it easy to import into a spreadsheet or plot with matplotlib.

---

## Plotting

```python
dimensions = list(dict.fromkeys(r['dimension'] for r in results))  # preserve order, deduplicate
for i, dim in enumerate(dimensions):
    subset = [r for r in results if r['dimension'] == dim]
    values = [r['value'] for r in subset]
    accs   = [r['accuracy'] for r in subset]
    axes[i].plot(values, accs, 'b-o')
```

One subplot per dimension. `dict.fromkeys(...)` is a Python trick to deduplicate a list while preserving insertion order (unlike `set()` which doesn't preserve order).
