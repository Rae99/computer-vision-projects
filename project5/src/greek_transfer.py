# Junrui Ding
# Project 5: Recognition using Deep Networks
# Task 3: Transfer learning to recognize Greek letters (alpha, beta, gamma)

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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


# [LEARN] Transfer learning in 3 lines:
# 1. load pretrained weights, 2. freeze them (requires_grad=False), 3. replace head.
# Only fc2 (153 params) trains — the rest of the network stays frozen.
# → learning/06-transfer-learning.md
# → learning/concepts/transfer-learning.md
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
    plt.savefig('greek_training_loss.png')
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


# Main function: builds model, trains on Greek letters, evaluates
def main(argv):
    # Path to folder containing alpha/, beta/, gamma/ subfolders
    training_set_path = argv[1] if len(argv) > 1 else 'data/greek_train'

    if not os.path.exists(training_set_path):
        print(f'Greek training data not found at: {training_set_path}')
        print('Usage: python greek_transfer.py <path_to_greek_data>')
        return

    greek_loader = get_greek_loader(training_set_path)
    model = build_greek_model()

    losses = train_greek(model, greek_loader, epochs=20)
    plot_greek_loss(losses)

    print('\nEvaluating on training set:')
    evaluate_greek(model, greek_loader)

    torch.save(model.state_dict(), 'greek_model.pth')
    print('Greek model saved to greek_model.pth')

    return


if __name__ == '__main__':
    main(sys.argv)
