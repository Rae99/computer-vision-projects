# Junrui Ding, Junyao Han
# Project 5: Recognition using Deep Networks
# Task 3: Transfer learning to recognize Greek letters (alpha, beta, gamma)

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from mnist_recognition import MyNetwork


# Transform to convert 133x133 color Greek letter images to 28x28 MNIST-like format
class GreekTransform:
    def __init__(self):
        pass

    # Converts RGB image to grayscale, scales, crops to 28x28, inverts to match MNIST
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# Loads Greek letter dataset from folder, returns DataLoader
def get_greek_loader(training_set_path, batch_size=5):
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return greek_train


def build_greek_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))

    # Freeze all parameters — gradients won't be computed for these during backprop
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer with 3-node output (alpha, beta, gamma)
    model.fc2 = nn.Linear(50, 3)
    # New layer has requires_grad=True by default

    print('Modified model for Greek letters:')
    print(model)
    return model


# Trains the Greek letter model, returns loss history
def train_greek(model, greek_loader, epochs=20):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in greek_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)

        avg_loss = epoch_loss / total
        acc = correct / total
        losses.append(avg_loss)
        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}')

    return losses


# Plots the training loss curve
def plot_greek_loss(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Greek Letters Transfer Learning - Training Loss')
    plt.savefig('../output/greek_training_loss.png')
    plt.show()


# Tests the model on a batch from the loader, prints predictions
def evaluate_greek(model, loader, class_names=['alpha', 'beta', 'gamma']):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                predicted_name = class_names[preds[i].item()]
                true_name = class_names[labels[i].item()]
                print(f'Predicted: {predicted_name:<10} True: {true_name}')
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f'\nOverall accuracy: {correct}/{total} = {correct/total:.4f}')


# Loads a single image, applies GreekTransform pipeline, returns tensor and processed numpy array
def preprocess_greek_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    # Resize to 133x133 to match expected input size
    img = cv2.resize(img, (133, 133))
    # Convert BGR to RGB for torchvision
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply transform pipeline manually
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0)
    # Get processed image for display (without normalize)
    display_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
    ])
    display_tensor = display_transform(pil_img)
    display_np = display_tensor.squeeze().numpy()
    return tensor, display_np


# Runs model on custom Greek letter images, prints and plots results
def test_custom_greek(model, folder, class_names=['alpha', 'beta', 'gamma']):
    image_files = sorted([f for f in os.listdir(folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f'No images found in {folder}')
        return

    results = []
    correct = 0
    for fname in image_files:
        # Determine true label from filename prefix
        true_label = None
        for i, name in enumerate(class_names):
            if fname.lower().startswith(name):
                true_label = i
                break
        if true_label is None:
            print(f'Could not determine label for {fname}, skipping')
            continue

        path = os.path.join(folder, fname)
        result = preprocess_greek_image(path)
        if result is None:
            continue
        tensor, display_np = result
        with torch.no_grad():
            output = model(tensor)
        pred = output.argmax().item()
        correct += int(pred == true_label)
        results.append((fname, display_np, pred, true_label))
        status = 'correct' if pred == true_label else 'WRONG'
        print(f'{fname}: Predicted={class_names[pred]:<8} True={class_names[true_label]:<8} {status}')

    print(f'\nAccuracy: {correct}/{len(results)} = {correct/len(results):.4f}')

    # Plot results
    n = len(results)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = np.array(axes).reshape(-1)
    for i, (fname, img, pred, true) in enumerate(results):
        color = 'green' if pred == true else 'red'
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'P:{class_names[pred]}\nT:{class_names[true]}', color=color, fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.suptitle('Custom Greek Letter Predictions')
    plt.tight_layout()
    plt.savefig('../output/custom_greek_results.png')
    plt.show()


# Main function: trains (or loads) Greek model, evaluates, optionally tests custom images
# Usage: python greek_transfer.py <training_path> [custom_test_path]
# If greek_model.pth exists, skips training and loads saved weights.
def main(argv):
    training_set_path = argv[1] if len(argv) > 1 else '../data/greek_train'
    custom_path = argv[2] if len(argv) > 2 else None

    if not os.path.exists(training_set_path):
        print(f'Greek training data not found at: {training_set_path}')
        print('Usage: python greek_transfer.py <training_path> [custom_test_path]')
        return

    greek_loader = get_greek_loader(training_set_path)

    model_path = 'greek_model.pth'
    if os.path.exists(model_path):
        print(f'Loading saved model from {model_path} (skipping training)')
        model = build_greek_model()
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model = build_greek_model()
        losses = train_greek(model, greek_loader, epochs=20)
        plot_greek_loss(losses)
        torch.save(model.state_dict(), model_path)
        print(f'Greek model saved to {model_path}')

    print('\nEvaluating on training set:')
    evaluate_greek(model, greek_loader)

    if custom_path and os.path.exists(custom_path):
        print(f'\nTesting on custom Greek images from {custom_path}:')
        test_custom_greek(model, custom_path)

    return


if __name__ == '__main__':
    main(sys.argv)
