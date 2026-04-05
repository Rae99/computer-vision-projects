# Junrui Ding
# Project 5: Recognition using Deep Networks
# Extension 4: Replace CNN conv1 with handcrafted Gabor filters, retrain remaining layers

import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mnist_recognition import MyNetwork, get_data, evaluate


# Generates 10 Gabor filters with varied orientations using cv2.getGaborKernel
# Returns a torch tensor of shape [10, 1, 5, 5]
def make_gabor_filters(num_filters=10, ksize=5, sigma=1.5, lambd=5.0, gamma=0.5, psi=0):
    filters = []
    for i in range(num_filters):
        # Vary orientation: theta = i * pi / num_filters
        theta = i * math.pi / num_filters
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=psi,
            ktype=cv2.CV_32F
        )
        filters.append(kernel)

    # Stack into tensor [10, 5, 5], then add channel dim → [10, 1, 5, 5]
    filters_np = np.stack(filters, axis=0)                         # [10, 5, 5]
    filters_tensor = torch.tensor(filters_np).unsqueeze(1)         # [10, 1, 5, 5]
    return filters_tensor


# Injects Gabor filters into model.conv1.weight and freezes conv1
def replace_conv1_with_gabor(model, gabor_filters):
    with torch.no_grad():
        model.conv1.weight.copy_(gabor_filters)

    # Freeze conv1 weights and bias so they are not updated during training
    model.conv1.weight.requires_grad = False
    if model.conv1.bias is not None:
        model.conv1.bias.requires_grad = False

    print('conv1 replaced with Gabor filters and frozen.')
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f'Frozen param groups: {frozen}, Trainable param groups: {trainable}')


# Trains only the unfrozen layers (conv2, fc1, fc2) for the given number of epochs
def train_with_gabor(model, train_loader, test_loader, epochs=5, lr=0.01):
    criterion = nn.NLLLoss()
    # Only pass parameters that require gradients to the optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.5
    )

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | '
              f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')

    return train_losses, test_losses, train_accs, test_accs


# Plots training/test loss and accuracy curves for the Gabor-initialized model
def plot_gabor_curves(train_losses, test_losses, train_accs, test_accs,
                      path='../output/ext4_gabor_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Gabor Network - Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, test_accs, 'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Gabor Network - Accuracy')
    ax2.legend()

    plt.suptitle('CNN with Frozen Gabor Conv1 (retraining conv2, fc1, fc2)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f'Training curves saved to {path}')


# Visualizes the 10 Gabor filters in a 2x5 grid
def visualize_gabor_filters(gabor_filters, path='../output/ext4_gabor_filters.png'):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    with torch.no_grad():
        for i, ax in enumerate(axes.flat):
            filt = gabor_filters[i, 0].numpy()  # [5, 5]
            ax.imshow(filt, cmap='viridis')
            theta_deg = i * 180 // 10
            ax.set_title(f'theta={theta_deg}deg')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Handcrafted Gabor Filters (10, varied orientations)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f'Gabor filter visualization saved to {path}')


# Loads the original mnist_model.pth and evaluates its test accuracy for comparison
def get_original_accuracy(test_loader):
    criterion = nn.NLLLoss()
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    _, acc = evaluate(model, test_loader, criterion)
    return acc


# Main function: loads MNIST model, replaces conv1 with Gabor, retrains, compares accuracy
def main(argv):
    print('Loading data...')
    train_loader, test_loader = get_data(batch_size=64)

    print('\nGenerating 10 Gabor filters (5x5, varied orientations)...')
    gabor_filters = make_gabor_filters(
        num_filters=10, ksize=5, sigma=1.5, lambd=5.0, gamma=0.5, psi=0
    )
    print(f'Gabor filter tensor shape: {gabor_filters.shape}')

    visualize_gabor_filters(gabor_filters)

    print('\nLoading pretrained mnist_model.pth...')
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth'))

    print('\nReplacing conv1 with Gabor filters and freezing conv1...')
    replace_conv1_with_gabor(model, gabor_filters)

    print('\nRetraining remaining layers (conv2, fc1, fc2) for 5 epochs...')
    train_losses, test_losses, train_accs, test_accs = train_with_gabor(
        model, train_loader, test_loader, epochs=5)

    plot_gabor_curves(train_losses, test_losses, train_accs, test_accs)

    gabor_final_acc = test_accs[-1]

    print('\nEvaluating original mnist_model.pth for comparison...')
    original_acc = get_original_accuracy(test_loader)

    print('\n--- Accuracy Comparison ---')
    print(f'Original MNIST model (fully trained CNN):  {original_acc:.4f}')
    print(f'Gabor conv1 model (conv2/fc retrained):    {gabor_final_acc:.4f}')
    diff = gabor_final_acc - original_acc
    direction = 'higher' if diff >= 0 else 'lower'
    print(f'Gabor model is {abs(diff):.4f} {direction} than the original model.')
    print()
    print('Observation: Gabor filters are hand-designed edge detectors that mimic')
    print('simple cells in visual cortex. By fixing conv1 to these filters, the')
    print('network loses flexibility in the first layer but gains interpretability.')
    print('If the Gabor model approaches the original accuracy, it suggests that')
    print('learned conv1 filters converge to Gabor-like patterns anyway (as is')
    print('commonly observed in CNNs trained on visual data).\n')

    return


if __name__ == '__main__':
    main(sys.argv)
