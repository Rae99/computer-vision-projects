# Junrui Ding
# Project 5: Recognition using Deep Networks
# Task 2: Examine the trained network - visualize filters and their effects

import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mnist_recognition import MyNetwork


# Loads the saved model from file
def load_model(path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Loads first training image from MNIST
def get_first_training_image():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
    image, label = train_set[0]
    return image, label


# Gets and prints the first layer filter weights
def get_first_layer_weights(model):
    weights = model.conv1.weight  # shape: [10, 1, 5, 5]
    print(f'Filter weights shape: {weights.shape}')
    print(f'(10 filters, 1 input channel, 5x5 each)\n')

    with torch.no_grad():
        for i in range(10):
            print(f'Filter {i}: {weights[i, 0].numpy()}')
            print()

    return weights


# Visualizes the 10 conv1 filters as a 3x4 grid (last 2 slots empty)
def visualize_filters(weights):
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    with torch.no_grad():
        for i in range(10):
            row, col = divmod(i, 4)
            filt = weights[i, 0].numpy()
            axes[row, col].imshow(filt, cmap='viridis')
            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        # Hide unused subplots
        for j in range(10, 12):
            row, col = divmod(j, 4)
            axes[row, col].axis('off')

    plt.suptitle('Conv1 Filter Weights')
    plt.tight_layout()
    plt.savefig('filters.png')
    plt.show()


# Applies all 10 conv1 filters to the first training image using cv2.filter2D
def apply_filters(model, image):
    # Convert image tensor to numpy array
    img_np = image.squeeze().numpy()

    weights = model.conv1.weight

    fig, axes = plt.subplots(5, 4, figsize=(10, 12))
    axes = axes.flat

    with torch.no_grad():
        for i in range(10):
            filt = weights[i, 0].numpy()

            # Apply filter using OpenCV filter2D
            filtered = cv2.filter2D(img_np, -1, filt)

            # Show original and filtered side by side
            axes[2 * i].imshow(img_np, cmap='gray')
            axes[2 * i].set_title(f'Original')
            axes[2 * i].set_xticks([])
            axes[2 * i].set_yticks([])

            axes[2 * i + 1].imshow(filtered, cmap='gray')
            axes[2 * i + 1].set_title(f'Filter {i}')
            axes[2 * i + 1].set_xticks([])
            axes[2 * i + 1].set_yticks([])

    plt.suptitle('Effect of Conv1 Filters on First Training Image')
    plt.tight_layout()
    plt.savefig('filter_effects.png')
    plt.show()


# Main function: loads model, analyzes first layer, applies filters
def main(argv):
    model = load_model('mnist_model.pth')
    print(model)
    print()

    weights = get_first_layer_weights(model)
    visualize_filters(weights)

    image, label = get_first_training_image()
    print(f'Applying filters to first training image (label: {label})')
    apply_filters(model, image)

    return


if __name__ == '__main__':
    main(sys.argv)
