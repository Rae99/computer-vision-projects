# Junrui Ding
# Project 5: Recognition using Deep Networks
# Task 5: Design your own experiment - ablation study on CNN architecture

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import csv

from mnist_recognition import get_data


# [LEARN] FlexibleNetwork = MyNetwork with 4 configurable hyperparameters.
# flat_size must match conv2_filters × 4 × 4 (derived from the spatial math).
# → learning/08-ablation-experiment.md
class FlexibleNetwork(nn.Module):
    # Builds CNN with configurable filter counts, hidden size, and dropout rate
    def __init__(self, conv1_filters=10, conv2_filters=20,
                 fc1_size=50, dropout_rate=0.5):
        super(FlexibleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size after two conv+pool layers on 28x28 input
        # After conv1(5x5) + pool(2x2): (28-4)/2 = 12
        # After conv2(5x5) + pool(2x2): (12-4)/2 = 4
        flat_size = conv2_filters * 4 * 4

        self.fc1 = nn.Linear(flat_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 10)
        self.flat_size = flat_size

    # Forward pass through the network
    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(-1, self.flat_size)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


# Trains a model for a fixed number of epochs, returns final test accuracy
def quick_train(model, train_loader, test_loader, epochs=3):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# Runs ablation study, varying one dimension at a time (linear search strategy)
def run_experiment(train_loader, test_loader):
    # Hypotheses:
    # 1. More conv filters → higher accuracy, up to a point (diminishing returns)
    # 2. Higher dropout → lower accuracy (too much regularization with limited data)
    # 3. Larger FC1 → marginal improvement (bottleneck is in conv layers for MNIST)

    results = []

    # Baseline
    baseline = {'conv1': 10, 'conv2': 20, 'fc1': 50, 'dropout': 0.5}
    print('Running ablation study...')
    print(f'{"Dimension":<20} {"Value":<10} {"Test Acc":<10}')
    print('-' * 42)

    # Dimension 1: Number of conv1 filters (hold others constant at baseline)
    conv1_options = [5, 10, 20, 40]
    best_conv1 = baseline['conv1']
    best_acc = 0
    for v in conv1_options:
        model = FlexibleNetwork(conv1_filters=v, conv2_filters=baseline['conv2'],
                                fc1_size=baseline['fc1'], dropout_rate=baseline['dropout'])
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'conv1_filters', 'value': v, 'accuracy': acc})
        print(f'{"conv1_filters":<20} {v:<10} {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_conv1 = v

    # Dimension 2: Dropout rate (use best conv1, hold others at baseline)
    dropout_options = [0.1, 0.25, 0.5, 0.75]
    best_dropout = baseline['dropout']
    best_acc = 0
    for v in dropout_options:
        model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=baseline['conv2'],
                                fc1_size=baseline['fc1'], dropout_rate=v)
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'dropout_rate', 'value': v, 'accuracy': acc})
        print(f'{"dropout_rate":<20} {v:<10} {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_dropout = v

    # Dimension 3: FC1 hidden size (use best conv1 and dropout)
    fc1_options = [25, 50, 100, 200]
    for v in fc1_options:
        model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=baseline['conv2'],
                                fc1_size=v, dropout_rate=best_dropout)
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'fc1_size', 'value': v, 'accuracy': acc})
        print(f'{"fc1_size":<20} {v:<10} {acc:.4f}')

    return results


# Saves experiment results to CSV
def save_results(results, path='experiment_results.csv'):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dimension', 'value', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults saved to {path}')


# Plots accuracy vs value for each dimension
def plot_results(results):
    dimensions = list(dict.fromkeys(r['dimension'] for r in results))
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5 * len(dimensions), 4))

    for i, dim in enumerate(dimensions):
        subset = [r for r in results if r['dimension'] == dim]
        values = [r['value'] for r in subset]
        accs = [r['accuracy'] for r in subset]
        axes[i].plot(values, accs, 'b-o')
        axes[i].set_xlabel(dim)
        axes[i].set_ylabel('Test Accuracy')
        axes[i].set_title(f'Effect of {dim}')

    plt.tight_layout()
    plt.savefig('experiment_results.png')
    plt.show()


# Main function: loads data, runs ablation study, saves and plots results
def main(argv):
    print('Loading data...')
    train_loader, test_loader = get_data(batch_size=64)

    results = run_experiment(train_loader, test_loader)
    save_results(results)
    plot_results(results)

    return


if __name__ == '__main__':
    main(sys.argv)
