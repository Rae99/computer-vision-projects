# Junrui Ding
# Project 5: Recognition using Deep Networks
# Extension 1: Expanded ablation study - adds conv2_filters and batch_size dimensions

import sys
import csv
import matplotlib.pyplot as plt

from mnist_recognition import get_data
from experiment import FlexibleNetwork, quick_train


# Runs expanded ablation study over 5 dimensions (conv1, dropout, fc1, conv2, batch_size)
# Uses greedy linear search: best value from each dimension carried forward
def run_experiment_more():
    # Hypotheses:
    # 4. More conv2 filters → higher accuracy (richer feature maps before FC)
    # 5. Smaller batch size → noisier but potentially better generalization
    #    Larger batch size → smoother gradients but may converge to sharper minima

    results = []

    # Baseline configuration
    baseline = {
        'conv1': 10,
        'conv2': 20,
        'fc1': 50,
        'dropout': 0.5,
        'batch_size': 64
    }

    print('Running expanded ablation study (5 dimensions)...')
    print(f'{"Dimension":<20} {"Value":<10} {"Test Acc":<10}')
    print('-' * 42)

    # Load data at baseline batch size for dimensions 1-4
    train_loader, test_loader = get_data(batch_size=baseline['batch_size'])

    # Dimension 1: conv1_filters
    conv1_options = [5, 10, 20, 40]
    best_conv1 = baseline['conv1']
    best_acc = 0.0
    for v in conv1_options:
        model = FlexibleNetwork(conv1_filters=v, conv2_filters=baseline['conv2'],
                                fc1_size=baseline['fc1'], dropout_rate=baseline['dropout'])
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'conv1_filters', 'value': v, 'accuracy': acc})
        print(f'{"conv1_filters":<20} {v:<10} {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_conv1 = v

    # Dimension 2: dropout_rate (carry forward best conv1)
    dropout_options = [0.1, 0.25, 0.5, 0.75]
    best_dropout = baseline['dropout']
    best_acc = 0.0
    for v in dropout_options:
        model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=baseline['conv2'],
                                fc1_size=baseline['fc1'], dropout_rate=v)
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'dropout_rate', 'value': v, 'accuracy': acc})
        print(f'{"dropout_rate":<20} {v:<10} {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_dropout = v

    # Dimension 3: fc1_size (carry forward best conv1 and dropout)
    fc1_options = [25, 50, 100, 200]
    best_fc1 = baseline['fc1']
    best_acc = 0.0
    for v in fc1_options:
        model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=baseline['conv2'],
                                fc1_size=v, dropout_rate=best_dropout)
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'fc1_size', 'value': v, 'accuracy': acc})
        print(f'{"fc1_size":<20} {v:<10} {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_fc1 = v

    # Dimension 4: conv2_filters (carry forward best conv1, dropout, fc1)
    conv2_options = [10, 20, 40, 80]
    best_conv2 = baseline['conv2']
    best_acc = 0.0
    for v in conv2_options:
        model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=v,
                                fc1_size=best_fc1, dropout_rate=best_dropout)
        acc = quick_train(model, train_loader, test_loader, epochs=3)
        results.append({'dimension': 'conv2_filters', 'value': v, 'accuracy': acc})
        print(f'{"conv2_filters":<20} {v:<10} {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_conv2 = v

    # Dimension 5: batch_size — reload data for each value
    batch_options = [32, 64, 128, 256]
    for v in batch_options:
        bs_train_loader, bs_test_loader = get_data(batch_size=v)
        model = FlexibleNetwork(conv1_filters=best_conv1, conv2_filters=best_conv2,
                                fc1_size=best_fc1, dropout_rate=best_dropout)
        acc = quick_train(model, bs_train_loader, bs_test_loader, epochs=3)
        results.append({'dimension': 'batch_size', 'value': v, 'accuracy': acc})
        print(f'{"batch_size":<20} {v:<10} {acc:.4f}')

    return results


# Saves experiment results to CSV
def save_results(results, path='../output/ext1_experiment_more.csv'):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dimension', 'value', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults saved to {path}')


# Plots accuracy vs value for each dimension in a row of subplots
def plot_results(results, path='../output/ext1_experiment_more.png'):
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

    plt.suptitle('Extended Ablation Study (5 Dimensions)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f'Plot saved to {path}')


# Main function: runs expanded ablation study, saves and plots results
def main(argv):
    results = run_experiment_more()
    save_results(results)
    plot_results(results)
    return


if __name__ == '__main__':
    main(sys.argv)
