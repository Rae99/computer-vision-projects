# Layer 4: Training Loop

**File:** `mnist_recognition.py` → `train_network()`, `evaluate()`

---

## The Core Loop

Training is a loop over epochs. Each epoch loops over batches. Each batch does four steps:

```python
for epoch in range(1, epochs + 1):
    model.train()                              # ← enable dropout, etc.
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()                  # 1. clear old gradients
        outputs = model(images)               # 2. forward pass
        loss = criterion(outputs, labels)     # 3. compute loss
        loss.backward()                       # 4. backpropagation
        optimizer.step()                      # 5. update weights
```

**Analogy:** Like a CI/CD pipeline that runs on every commit (batch):
1. Zero out old gradient accumulations (like clearing a build cache)
2. Run the model (like building the app)
3. Measure how wrong the output is (like running tests)
4. Compute which direction to move each weight (like the test reporter showing diffs)
5. Actually move the weights (like applying the fix)

---

## Step-by-Step

### 1. `optimizer.zero_grad()`
PyTorch **accumulates** gradients by default. If you don't clear them, gradients from batch N add to batch N+1, causing incorrect updates. Always call this at the start of each batch.

### 2. `outputs = model(images)`
Runs `forward()` — data flows from pixels to log-probabilities. Returns shape `[64, 10]` (64 images × 10 class scores).

### 3. `loss = criterion(outputs, labels)`
`criterion` is `nn.NLLLoss()` — measures how wrong the predictions are. Returns a single scalar. Lower = better.

### 4. `loss.backward()`
Backpropagation: computes the gradient of the loss with respect to every weight in the network. This is the "chain rule" applied automatically through all layers. PyTorch builds a computation graph during the forward pass and walks it backward here.

### 5. `optimizer.step()`
The optimizer (`SGD`) uses the gradients to nudge every weight slightly in the direction that reduces loss:
```
weight = weight - learning_rate * gradient
```

---

## The Optimizer: SGD

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

- **SGD** = Stochastic Gradient Descent. "Stochastic" because we use mini-batches, not the full dataset.
- **lr=0.01** — learning rate: how big each weight update is. Too high → overshoots. Too low → trains slowly.
- **momentum=0.5** — like inertia: carries some velocity from the previous step. Helps escape shallow local minima.

The transformer (Layer 7) uses **Adam** instead (`lr=1e-3`). Adam adapts the learning rate per-parameter and usually converges faster.

---

## evaluate() — Inference Mode

```python
def evaluate(model, loader, criterion):
    model.eval()                  # ← disable dropout
    with torch.no_grad():         # ← don't track gradients (saves memory)
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            pred = outputs.argmax(dim=1)  # pick the highest-scoring class
            correct += (pred == labels).sum().item()
```

Two key differences from training:
- `model.eval()` — disables dropout (all neurons active), makes predictions deterministic
- `torch.no_grad()` — tells PyTorch not to build a computation graph (no `.backward()` will be called), saving memory and compute

---

## Loss: NLLLoss

`nn.NLLLoss()` expects **log-probabilities** as input (which is why `forward()` ends with `log_softmax`). It computes:

```
loss = -output[correct_class]
```

The more confident the model is about the right answer (output[correct_class] close to 0 in log-space), the lower the loss. See `concepts/log-softmax-nllloss.md` for the full picture.

---

## Training vs. Test Loss

After each epoch, both are evaluated:
```python
train_loss, train_acc = evaluate(model, train_loader, criterion)
test_loss, test_acc   = evaluate(model, test_loader, criterion)
```

- **Train loss going down** — the model is learning
- **Test loss also going down** — the model is generalizing (not just memorizing)
- **Train loss down, test loss up** — overfitting: the model memorized training data but can't generalize

---

## Deep Dives

- → `concepts/training-loop.md` (forward pass, loss landscape, gradient descent)
- → `concepts/log-softmax-nllloss.md` (why these two are paired)
