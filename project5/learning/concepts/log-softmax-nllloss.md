# Concept: log_softmax + NLLLoss

## Why These Two Are Always Paired

In `MyNetwork.forward()`, the last line is:
```python
x = torch.log_softmax(self.fc2(x), dim=1)
```
And the loss is:
```python
criterion = nn.NLLLoss()
```

These two must be used together. Here's why.

---

## Softmax: Raw Scores → Probabilities

`fc2` outputs 10 raw numbers (called **logits**), one per digit class. These could be anything: `-3.2, 7.1, 0.5, ...`. They don't add up to 1 and can't be interpreted as probabilities.

Softmax converts them to probabilities:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

Now all 10 values are in (0, 1) and sum to 1 — a proper probability distribution.

---

## log_softmax: Numerically Stable Version

```python
log_softmax(x_i) = log(softmax(x_i)) = x_i - log(sum(exp(x_j)))
```

**Why log?** Two reasons:
1. **Numerical stability**: `exp()` of large numbers overflows. Computing log(softmax) directly avoids ever materializing the huge `exp()` values.
2. **NLLLoss expects log-probabilities**: the loss function is designed to work with logs.

---

## NLLLoss: Negative Log-Likelihood

```python
loss = -output[batch_idx, correct_label]
```

It simply picks out the log-probability the model assigned to the correct class, and negates it.

- If model says: log P("3") = -0.01 (very confident) → loss = 0.01 ✓
- If model says: log P("3") = -5.0 (not confident) → loss = 5.0 ✗

Combined with log_softmax: `NLLLoss(log_softmax(x))` = **cross-entropy loss**.

---

## Code Before/After

```python
# WRONG: using softmax + NLLLoss (double-log problem)
x = torch.softmax(self.fc2(x), dim=1)
criterion = nn.NLLLoss()  # NLLLoss expects log-probabilities, not probabilities

# CORRECT option 1: log_softmax + NLLLoss (used in this project)
x = torch.log_softmax(self.fc2(x), dim=1)
criterion = nn.NLLLoss()

# CORRECT option 2: raw logits + CrossEntropyLoss (equivalent, more common in practice)
# x = self.fc2(x)  # no activation
# criterion = nn.CrossEntropyLoss()  # applies log_softmax internally
```

---

## Where It Shows Up

| Usage | File |
|-------|------|
| Output of CNN | `mnist_recognition.py:32` |
| Output of transformer | `mnist_transformer.py:74` |
| Output of FlexibleNetwork | `experiment.py:43` |
| Loss criterion | All training files |
