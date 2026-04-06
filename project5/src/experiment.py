# Junrui Ding, Junyao Han
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
# Total variants: 8 + 8 + 8 + 8 + 7 + 7 + 7 + 7 = 60 configurations
def run_experiment(train_loader, test_loader):
    # Hypotheses:
    # 1. conv1_filters: more filters → higher accuracy up to a point (diminishing returns)
    # 2. conv2_filters: same as above, larger effect since conv2 feeds FC layers
    # 3. dropout_rate: moderate dropout helps; too high hurts
    # 4. fc1_size: optimal middle ground; too small underfits, too large overfits in 3 epochs
    # 5. batch_size: smaller batches → noisier but often better generalization
    # 6. learning_rate: too low = slow convergence; too high = unstable

    results = []
    baseline = {'conv1': 10, 'conv2': 20, 'fc1': 50, 'dropout': 0.5, 'lr': 0.01}
    print('Running ablation study (60 variants)...')
    print(f'{"Dimension":<22} {"Value":<10} {"Test Acc":<10}')
    print('-' * 44)

    def run_dim(dim_name, options, get_model_fn, get_loader_fn=None):
        best_val = options[0]
        best_acc = 0
        for v in options:
            loader = get_loader_fn(v) if get_loader_fn else (train_loader, test_loader)
            model = get_model_fn(v)
            acc = quick_train(model, loader[0], loader[1], epochs=3)
            results.append({'dimension': dim_name, 'value': v, 'accuracy': acc})
            print(f'{dim_name:<22} {str(v):<10} {acc:.4f}')
            if acc > best_acc:
                best_acc = acc
                best_val = v
        return best_val

    # Dimension 1: conv1_filters
    best_conv1 = run_dim('conv1_filters', [4, 8, 10, 16, 20, 32, 40, 64],
        lambda v: FlexibleNetwork(conv1_filters=v, conv2_filters=baseline['conv2'],
                                  fc1_size=baseline['fc1'], dropout_rate=baseline['dropout']))

    # Dimension 2: conv2_filters (use best conv1)
    best_conv2 = run_dim('conv2_filters', [10, 16, 20, 32, 40, 64, 80],
        lambda v: FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=v,
                                  fc1_size=baseline['fc1'], dropout_rate=baseline['dropout']))

    # Dimension 3: dropout_rate (use best conv1, conv2)
    best_dropout = run_dim('dropout_rate', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
        lambda v: FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=best_conv2,
                                  fc1_size=baseline['fc1'], dropout_rate=v))

    # Dimension 4: fc1_size (use best conv1, conv2, dropout)
    best_fc1 = run_dim('fc1_size', [16, 25, 50, 75, 100, 150, 200, 256],
        lambda v: FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=best_conv2,
                                  fc1_size=v, dropout_rate=best_dropout))

    # Dimension 5: batch_size (use best arch params)
    from mnist_recognition import get_data as _get_data

    def run_batch_dim():
        best_val = 64
        best_acc = 0
        for v in [32, 48, 64, 96, 128, 192, 256]:
            tr, te = _get_data(batch_size=v)
            model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=best_conv2,
                                    fc1_size=best_fc1, dropout_rate=best_dropout)
            acc = quick_train(model, tr, te, epochs=3)
            results.append({'dimension': 'batch_size', 'value': v, 'accuracy': acc})
            print(f'{"batch_size":<22} {str(v):<10} {acc:.4f}')
            if acc > best_acc:
                best_acc = acc
                best_val = v
        return best_val

    best_batch = run_batch_dim()

    # Dimension 6: learning_rate (use best everything)
    tr_final, te_final = _get_data(batch_size=best_batch)

    def run_lr_dim():
        for v in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=best_conv2,
                                    fc1_size=best_fc1, dropout_rate=best_dropout)
            criterion = nn.NLLLoss()
            optimizer = optim.SGD(model.parameters(), lr=v, momentum=0.5)
            for _ in range(3):
                model.train()
                for images, labels in tr_final:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in te_final:
                    preds = model(images).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            results.append({'dimension': 'learning_rate', 'value': v, 'accuracy': acc})
            print(f'{"learning_rate":<22} {str(v):<10} {acc:.4f}')

    run_lr_dim()

    print(f'\nTotal variants evaluated: {len(results)}')
    return results


# Saves experiment results to CSV
def save_results(results, path='../output/experiment_results.csv'):
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
    plt.savefig('../output/experiment_results.png')
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
