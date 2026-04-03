# Junrui Ding
# Project 5: Recognition using Deep Networks
# Task 4: Re-implement MNIST recognition using Transformer layers

import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from mnist_recognition import get_data, evaluate


# [LEARN] ViT: treat the image as a sequence of patches (like words in a sentence).
# 28×28 / 7×7 = 16 patches. Each patch → a 64-dim token → transformer → classify.
# → learning/07-vision-transformer.md
# → learning/concepts/vision-transformer.md
class NetTransformer(nn.Module):
    # Builds transformer network: patch embedding -> transformer encoder -> classifier
    def __init__(self, image_size=28, patch_size=7, num_classes=10,
                 d_model=64, nhead=4, num_layers=2, mlp_dim=128, dropout=0.1):
        super(NetTransformer, self).__init__()

        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 4x4 = 16 patches for 28x28, patch=7
        self.patch_dim = patch_size * patch_size            # 7x7 = 49 pixels per patch

        # Linear layer to embed each patch into d_model dimensions
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # Positional embedding: learned position info for each patch
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: linear -> relu -> output
        self.classifier = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    # Divides image into patches, embeds them, runs through transformer, classifies
    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Divide image into patches: B x num_patches x patch_dim
        x = x.unfold(2, p, p).unfold(3, p, p)          # B x 1 x (H/p) x (W/p) x p x p
        x = x.contiguous().view(B, -1, p * p)           # B x num_patches x patch_dim

        # Embed patches
        x = self.patch_embedding(x)                     # B x num_patches x d_model

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer encoder
        x = self.transformer(x)                         # B x num_patches x d_model

        # Average all patch tokens to get single representation
        x = x.mean(dim=1)                               # B x d_model

        # Classify
        x = self.classifier(x)                         # B x num_classes
        x = torch.log_softmax(x, dim=1)
        return x


# Trains the transformer model for the given number of epochs
def train_transformer(model, train_loader, test_loader, epochs=5, lr=1e-3):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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


# Plots training and test curves for the transformer model
def plot_transformer_curves(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Transformer - Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, test_accs, 'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Transformer - Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('transformer_curves.png')
    plt.show()


# Main function: builds and trains the transformer model
def main(argv):
    train_loader, test_loader = get_data(batch_size=64)

    model = NetTransformer(
        image_size=28,
        patch_size=7,
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        mlp_dim=128,
        dropout=0.1
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')

    train_losses, test_losses, train_accs, test_accs = train_transformer(
        model, train_loader, test_loader, epochs=5)

    plot_transformer_curves(train_losses, test_losses, train_accs, test_accs)
    torch.save(model.state_dict(), 'transformer_model.pth')
    print('Transformer model saved.')

    return


if __name__ == '__main__':
    main(sys.argv)
