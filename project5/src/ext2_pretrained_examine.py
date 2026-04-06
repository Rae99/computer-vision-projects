# Junrui Ding
# Project 5: Recognition using Deep Networks
# Extension 2: Examine pretrained ResNet18 filters and compare to MNIST CNN filters

import sys
import torch
import torchvision
import torchvision.models
import matplotlib.pyplot as plt
import numpy as np

from mnist_recognition import MyNetwork
from mnist_examine import visualize_filters


# Loads ResNet18 with pretrained ImageNet weights
def load_resnet18():
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.eval()
    return model


# Extracts conv1 weights from ResNet18: shape [64, 3, 7, 7]
def get_resnet18_conv1_weights(model):
    weights = model.conv1.weight  # [64, 3, 7, 7]
    print(f'ResNet18 conv1 weight shape: {weights.shape}')
    print('(64 filters, 3 RGB input channels, 7x7 each)\n')
    return weights


# Visualizes all 64 ResNet18 conv1 filters in an 8x8 grid
# Uses mean across the 3 RGB channels to produce a 7x7 grayscale image per filter
def visualize_resnet18_filters(weights, path='../output/ext2_resnet18_filters.png'):
    fig, axes = plt.subplots(8, 8, figsize=(14, 14))
    with torch.no_grad():
        for i in range(64):
            row, col = divmod(i, 8)
            # Average across RGB channels: shape [7, 7]
            filt = weights[i].mean(dim=0).numpy()
            axes[row, col].imshow(filt, cmap='viridis')
            axes[row, col].set_title(f'{i}', fontsize=6)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.suptitle('ResNet18 Conv1 Filters (64 filters, mean of RGB channels)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f'ResNet18 filter visualization saved to {path}')


# Loads the trained MNIST CNN model from file
def load_mnist_model(path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Prints observations comparing MNIST conv1 filters to ResNet18 conv1 filters
def print_comparison():
    print('\n--- Observations: MNIST CNN vs ResNet18 Conv1 Filters ---')
    print('MNIST conv1 (10 filters, 1 channel, 5x5):')
    print('  - Small filters (5x5) suited for small 28x28 grayscale digits')
    print('  - Only 10 filters; limited diversity of learned patterns')
    print('  - Filters tend to detect simple edges and blobs in grayscale')
    print('  - Trained on one domain (handwritten digits), so filters are narrow')
    print()
    print('ResNet18 conv1 (64 filters, 3 channels, 7x7):')
    print('  - Larger filters (7x7) capture broader spatial patterns')
    print('  - 64 filters provide rich, diverse feature detectors')
    print('  - Filters show oriented edge detectors, color-opponent patterns,')
    print('    and Gabor-like structures — resembling V1 receptive fields')
    print('  - Trained on ImageNet (1000 classes, natural images): far more diverse')
    print('  - RGB-channel structure means filters encode color contrast as well')
    print()
    print('Key takeaway: ResNet18 filters are more structured and interpretable,')
    print('reflecting Gabor-like orientation selectivity from large-scale training.')
    print('MNIST filters are simpler because the task (10 digit classes, grayscale)')
    print('requires far less representational diversity.\n')


# Main function: loads ResNet18 and MNIST models, visualizes and compares filters
def main(argv):
    print('Loading ResNet18 pretrained model...')
    resnet = load_resnet18()
    print(resnet)
    print()

    resnet_weights = get_resnet18_conv1_weights(resnet)
    visualize_resnet18_filters(resnet_weights)

    print('Loading trained MNIST CNN model...')
    mnist_model = load_mnist_model('mnist_model.pth')
    mnist_weights = mnist_model.conv1.weight  # [10, 1, 5, 5]
    print(f'MNIST conv1 weight shape: {mnist_weights.shape}')
    print('(10 filters, 1 input channel, 5x5 each)\n')

    # Reuse visualize_filters from mnist_examine (saves to ../output/filters.png)
    print('Visualizing MNIST conv1 filters (reusing mnist_examine.visualize_filters)...')
    visualize_filters(mnist_weights)

    print_comparison()
    return


if __name__ == '__main__':
    main(sys.argv)
