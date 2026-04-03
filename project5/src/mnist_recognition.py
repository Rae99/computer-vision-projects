# Junrui Ding
# Project 5: Recognition using Deep Networks
# Task 1: Build and train a CNN to recognize MNIST digits

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class MyNetwork(nn.Module):
    # Builds the network layers as specified in the project
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)       # 10 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)      # 20 5x5 filters
        self.dropout = nn.Dropout(p=0.5)                    # 50% dropout
        self.pool = nn.MaxPool2d(2, 2)                      # 2x2 max pool
        self.fc1 = nn.Linear(320, 50)                       # fully connected, 50 nodes
        self.fc2 = nn.Linear(50, 10)                        # output layer, 10 nodes

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))            # conv1 -> pool -> relu
        x = torch.relu(self.pool(self.dropout(self.conv2(x))))  # conv2 -> dropout -> pool -> relu
        x = x.view(-1, 320)                                 # flatten
        x = torch.relu(self.fc1(x))                         # fc1 -> relu
        x = torch.log_softmax(self.fc2(x), dim=1)          # fc2 -> log_softmax
        return x


def get_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Plots the first 6 images from the test set
def plot_first_six(test_loader):
    images, labels = next(iter(test_loader))
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('First 6 Test Set Examples')
    plt.tight_layout()
    plt.savefig('first_six.png')
    plt.show()


# Evaluates the model on a data loader, returns average loss and accuracy
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_network(model, train_loader, test_loader, epochs=5, lr=0.01):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate after each epoch
        train_loss, train_acc = evaluate(model, train_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | '
              f'Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies


# Plots training and test loss and accuracy curves
def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()

    ax2.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


# Saves the trained model weights to a file
def save_model(model, path='mnist_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


# Main function: runs the full training pipeline
def main(argv):
    batch_size = 64
    epochs = 5

    train_loader, test_loader = get_data(batch_size)
    plot_first_six(test_loader)

    model = MyNetwork()
    print(model)

    train_losses, test_losses, train_accs, test_accs = train_network(
        model, train_loader, test_loader, epochs=epochs)

    plot_training_curves(train_losses, test_losses, train_accs, test_accs)
    save_model(model)

    return


if __name__ == '__main__':
    main(sys.argv)
