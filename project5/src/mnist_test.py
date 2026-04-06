# Junrui Ding, Junyao Han
# Project 5: Recognition using Deep Networks
# Task 1E-F: Load trained model, run on test set and custom handwritten digits

import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from mnist_recognition import MyNetwork


# Loads the saved model from file
def load_model(path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Loads MNIST test set without shuffling
def get_test_loader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)
    return test_loader


# Runs the model on the first 10 test examples and prints results
def run_on_first_ten(model, test_loader):
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images)

    print(f'{"Index":<6} {"Output Values":<60} {"Predicted":<10} {"Correct":<10}')
    print('-' * 90)
    for i in range(10):
        output_vals = [f'{v:.2f}' for v in outputs[i].tolist()]
        predicted = outputs[i].argmax().item()
        correct = labels[i].item()
        print(f'{i:<6} {str(output_vals):<60} {predicted:<10} {correct:<10}')

    return images, labels, outputs


# Plots the first 9 test images in a 3x3 grid with predictions above each
def plot_first_nine(images, labels, outputs):
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i, ax in enumerate(axes.flat):
        prediction = outputs[i].argmax().item()
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Prediction: {prediction}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('../output/test_predictions.png')
    plt.show()


# Loads a single handwritten digit image, converts to 28x28 grayscale tensor
# matching MNIST format (white digit on black background)
def preprocess_custom_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f'Could not read image: {path}')
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))

    # MNIST digits are white on black; if your photo is black on white, invert
    # Check: if the background is bright (high mean), invert
    if resized.mean() > 127:
        resized = cv2.bitwise_not(resized)

    # Normalize same as MNIST: divide by 255, then normalize with mean/std
    tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # shape: 1x1x28x28
    return tensor, resized


# Runs the model on all custom handwritten digit images in a folder
def test_custom_digits(model, folder='my_digits', save_name='custom_digits_results.png'):
    if not os.path.exists(folder):
        print(f'Folder "{folder}" not found. Create it and put your digit images there.')
        return

    image_files = sorted([f for f in os.listdir(folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f'No images found in "{folder}"')
        return

    results = []
    for fname in image_files:
        path = os.path.join(folder, fname)
        result = preprocess_custom_image(path)
        if result is None:
            continue
        tensor, raw = result
        with torch.no_grad():
            output = model(tensor)
        prediction = output.argmax().item()
        results.append((fname, raw, prediction))
        print(f'{fname}: Predicted = {prediction}')

    # Plot results
    n = len(results)
    if n == 0:
        return
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    axes = np.array(axes).reshape(-1)
    for i, (fname, img, pred) in enumerate(results):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Pred: {pred}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.suptitle('Custom Handwritten Digits')
    plt.tight_layout()
    plt.savefig(f'../output/{save_name}')
    plt.show()


# Main function: loads model, runs on test set, runs on custom digits
def main(argv):
    model = load_model('mnist_model.pth')
    test_loader = get_test_loader()

    images, labels, outputs = run_on_first_ten(model, test_loader)
    plot_first_nine(images, labels, outputs)

    test_custom_digits(model, folder='my_digits', save_name='custom_digits_v1.png')
    test_custom_digits(model, folder='my_digits_v2', save_name='custom_digits_v2.png')
    return


if __name__ == '__main__':
    main(sys.argv)
