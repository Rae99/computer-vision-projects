# Concept: Training Loop

## Plain English

Training is how the network goes from random weights to useful weights. It's a cycle of making a prediction, measuring how wrong it was, and adjusting the weights to be less wrong next time.

---

## The Loss Landscape Analogy

Imagine all possible weight configurations as a hilly landscape. The height of a point = loss (how wrong the model is). Training is like being blindfolded on this landscape and trying to walk downhill:
- You can only feel the slope at your current position (the gradient)
- You take small steps in the downhill direction (gradient descent)
- Eventually you reach a valley (low loss = good model)

---

## The Four-Step Cycle

```python
optimizer.zero_grad()          # start fresh — clear slope readings from last step
outputs = model(images)        # feel where you are (forward pass)
loss = criterion(outputs, labels) # measure how high up you are (compute loss)
loss.backward()                # compute which direction is downhill (backprop)
optimizer.step()               # take a step downhill (update weights)
```

---

## Backpropagation (the chain rule)

`loss.backward()` computes `∂loss/∂weight` for every single weight in the network — thousands of them — automatically.

How? The forward pass builds a **computation graph** — a record of every operation done to the data. Backprop walks this graph in reverse, applying the chain rule at each step.

**Analogy:** Like `git blame` for the loss. "The model was wrong because fc2 neuron 3 was too confident, because fc1 produced a large input, because conv2 filter 7 activated strongly on a certain region..." Backprop traces this chain back to every weight.

---

## Code Before/After: Learning Rate Effect

```python
# Too high: overshoots — loss bounces around
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Too low: trains correctly but very slowly
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Goldilocks for this model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

---

## SGD vs. Adam

| | SGD | Adam |
|--|-----|------|
| Update rule | `w -= lr * grad` | Adapts lr per-parameter based on gradient history |
| Momentum | Optional (`momentum=0.5`) | Built-in (beta1, beta2) |
| Used in | CNN training, Greek transfer | Vision Transformer |
| Hyperparameter sensitivity | Higher | Lower |

Adam is generally more forgiving — it adjusts the effective learning rate for each weight automatically. SGD requires more careful tuning but often generalizes better when tuned well.

---

## Where It Shows Up

| Usage | File | Function |
|-------|------|----------|
| CNN training | `mnist_recognition.py` | `train_network()` |
| Greek letter fine-tuning | `greek_transfer.py` | `train_greek()` |
| Transformer training | `mnist_transformer.py` | `train_transformer()` |
| Quick ablation training | `experiment.py` | `quick_train()` |
