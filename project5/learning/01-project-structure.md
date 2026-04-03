# Layer 1: Project Structure & Entry Points

## The Big Picture

This project has 6 Python files, each handling one task from the assignment. They're not a web app — there's no server, no routes — but the structural pattern should feel familiar.

```
mnist_recognition.py   ← defines MyNetwork + trains it  (like a "model + controller")
mnist_test.py          ← loads saved model, runs inference  (like a "read-only API endpoint")
mnist_examine.py       ← visualizes internal weights  (like a "debug/introspection tool")
greek_transfer.py      ← adapts the model to a new task  (like "forking a component")
mnist_transformer.py   ← alternative architecture  (like "rewriting in a different framework")
experiment.py          ← systematic testing  (like "A/B testing your architecture")
```

---

## The Python Entry Point Pattern

Every file ends with:

```python
if __name__ == '__main__':
    main(sys.argv)
```

**Web dev analogy:** This is like `if (require.main === module)` in Node.js. It means:
- When you run `python mnist_recognition.py` directly → `main()` is called
- When another file does `from mnist_recognition import MyNetwork` → `main()` is **not** called

This is why `mnist_test.py` can import `MyNetwork` from `mnist_recognition.py` without accidentally training the network again.

---

## Module Imports Between Files

Three files import from `mnist_recognition.py`:

```python
# mnist_test.py
from mnist_recognition import MyNetwork

# mnist_examine.py
from mnist_recognition import MyNetwork

# mnist_transformer.py
from mnist_recognition import get_data, evaluate

# experiment.py
from mnist_recognition import get_data
```

**Web dev analogy:** This is just like ES module imports (`import { MyNetwork } from './mnist_recognition'`). Python uses the filename as the module name.

---

## Function-First Structure

Each file follows this pattern:
1. Imports
2. Class definitions (the model)
3. Helper functions (`get_data`, `train_network`, `plot_*`, `save_*`)
4. `main()` — wires everything together
5. `if __name__ == '__main__': main(sys.argv)`

This is the professor's required structure. Think of `main()` like `app.listen()` in Express — it's the orchestrator that calls everything in order.

---

## What `sys.argv` Is

`sys.argv` is a list of command-line arguments, like `process.argv` in Node.

```python
# If you run: python greek_transfer.py data/greek_train/
# Then sys.argv = ['greek_transfer.py', 'data/greek_train/']
training_set_path = argv[1] if len(argv) > 1 else 'data/greek_train'
```

---

## Key Files to Read Next

- **Layer 2** starts in `mnist_recognition.py` → `get_data()` function
- The saved model file `mnist_model.pth` is produced by `mnist_recognition.py` and consumed by the other files — it's the "database" that persists trained weights between runs
