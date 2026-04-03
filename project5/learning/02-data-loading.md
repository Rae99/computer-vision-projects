# Layer 2: Data Loading & Preprocessing

**Files:** `mnist_recognition.py` → `get_data()`, `mnist_test.py` → `get_test_loader()`, `preprocess_custom_image()`

---

## The MNIST Dataset

MNIST is 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels.
- 60,000 for training, 10,000 for testing
- Each image is a 2D grid of pixel values 0–255

**Analogy:** Think of it like a database of 70k records, each with one "column" — a 28×28 pixel grid — and a label (the digit).

---

## How PyTorch Loads Data: Three Layers

```python
# mnist_recognition.py, get_data()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
```

### Layer 1: The Dataset (`MNIST`)
Downloads and stores the raw images. Like a database table — it knows how to get item N.

### Layer 2: The Transform (`transforms.Compose`)
A pipeline applied to each image when it's fetched. Like Express middleware — each transform modifies `x` and passes it along.

- `ToTensor()` — converts the PIL image to a float tensor, scaling 0–255 → 0.0–1.0
- `Normalize((0.1307,), (0.3081,))` — subtracts mean, divides by std → see `concepts/normalization.md`

### Layer 3: The DataLoader
Wraps the dataset, batches it, optionally shuffles it. Like paginating an API:
- `batch_size=64` → fetch 64 images at a time
- `shuffle=True` → randomize order each epoch (good for training)
- `shuffle=False` → keep order (important for test set — so first 10 are always the same)

---

## What a Batch Looks Like

```python
images, labels = next(iter(train_loader))
# images.shape → torch.Size([64, 1, 28, 28])
#   64 images, 1 color channel (grayscale), 28 rows, 28 cols
# labels.shape → torch.Size([64])
#   one integer per image: 0-9
```

**Analogy:** Like getting a JSON response `[{ image: <28x28 array>, label: 7 }, ...]` for 64 items.

---

## Custom Image Preprocessing (`mnist_test.py`)

When you take a photo of a handwritten digit, it needs to be transformed to match MNIST format:

```python
def preprocess_custom_image(path):
    img = cv2.imread(path)               # load as BGR color image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # → grayscale
    resized = cv2.resize(gray, (28, 28)) # → 28x28

    # MNIST digits are WHITE on BLACK background
    # Your photo is probably BLACK on WHITE → invert it
    if resized.mean() > 127:             # if background is bright
        resized = cv2.bitwise_not(resized)

    # Normalize same as MNIST training data
    tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # → shape [1, 1, 28, 28]
```

The auto-inversion check (`mean > 127`) is clever: if the average pixel is bright, the image is probably light-background/dark-digit — the opposite of MNIST — so flip it.

---

## The GreekTransform (`greek_transfer.py`)

Greek letter images are 133×133 color photos that need to become 28×28 grayscale:

```python
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)  # color → gray
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)  # scale down
        x = torchvision.transforms.functional.center_crop(x, (28, 28))  # crop center
        return torchvision.transforms.functional.invert(x)  # white ↔ black
```

The `36/128` scale factor shrinks the letter so it fits within a 28×28 crop. The invert makes Greek letter images match MNIST's white-on-black convention.

---

## Deep Dives

- → `concepts/normalization.md` (why 0.1307 and 0.3081?)
