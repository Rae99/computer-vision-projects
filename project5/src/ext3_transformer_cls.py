# Junrui Ding, Junyao Han
# Project 5: Recognition using Deep Networks
# Extension 3: Vision Transformer with CLS token classification (instead of mean pooling)

import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mnist_recognition import get_data, evaluate


class NetTransformerCLS(nn.Module):
    # Transformer network that prepends a learnable CLS token and uses it for classification
    # Modified from NetTransformer: replaces mean pooling with CLS token (index 0)
    def __init__(self, image_size=28, patch_size=7, num_classes=10,
                 d_model=64, nhead=4, num_layers=4, mlp_dim=128, dropout=0.1):
        super(NetTransformerCLS, self).__init__()

        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 4x4 = 16 patches for 28x28, patch=7
        self.patch_dim = patch_size * patch_size            # 7x7 = 49 pixels per patch

        # Linear layer to embed each patch into d_model dimensions
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)

        # Learnable CLS token: prepended to the patch sequence before positional embedding
        # Shape: (1, 1, d_model) — broadcast over the batch dimension
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding: one entry per patch PLUS one for the CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        # Transformer encoder (deeper than baseline: num_layers=4 instead of 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head applied to the CLS token output only
        self.classifier = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    # Divides image into patches, prepends CLS token, runs transformer, classifies via CLS output
    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Divide image into non-overlapping patches: B x num_patches x patch_dim
        x = x.unfold(2, p, p).unfold(3, p, p)      # B x 1 x (H/p) x (W/p) x p x p
        x = x.contiguous().view(B, -1, p * p)       # B x num_patches x patch_dim

        # Embed patches into d_model dimensions
        x = self.patch_embedding(x)                 # B x num_patches x d_model

        # Expand CLS token to match batch size and prepend to patch sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)   # B x 1 x d_model
        x = torch.cat([cls_tokens, x], dim=1)            # B x (num_patches+1) x d_model

        # Add positional embedding to all tokens (CLS + patches)
        x = x + self.pos_embedding                       # B x (num_patches+1) x d_model

        # Run through transformer encoder
        x = self.transformer(x)                          # B x (num_patches+1) x d_model

        # Use only the CLS token output (index 0) for classification
        cls_out = x[:, 0, :]                             # B x d_model

        # Classify
        x = self.classifier(cls_out)                     # B x num_classes
        x = torch.log_softmax(x, dim=1)
        return x


# Trains the CLS transformer model for the given number of epochs
def train_transformer_cls(model, train_loader, test_loader, epochs=5, lr=1e-3):
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


# Plots training and test loss/accuracy curves for the CLS transformer
def plot_cls_curves(train_losses, test_losses, train_accs, test_accs,
                    path='../output/ext3_transformer_cls_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Transformer CLS - Loss')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, test_accs, 'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Transformer CLS - Accuracy')
    ax2.legend()

    plt.suptitle('Transformer with CLS Token (4 layers)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f'Curves saved to {path}')


# Prints comparison between CLS token and mean pooling strategies
def print_comparison_note(test_accs):
    final_acc = test_accs[-1] if test_accs else float('nan')
    print('\n--- CLS Token vs Mean Pooling ---')
    print(f'CLS token final test accuracy (4 layers): {final_acc:.4f}')
    print()
    print('CLS token approach (this script):')
    print('  - A single learnable vector is prepended to the patch sequence')
    print('  - Attends to all patches via self-attention throughout the encoder')
    print('  - Only the CLS output is used to classify; patch tokens are discarded')
    print('  - Follows BERT / ViT convention; natural for classification tasks')
    print()
    print('Mean pooling approach (mnist_transformer.py):')
    print('  - All patch tokens are averaged after the transformer')
    print('  - Simpler; no extra parameter; treats all positions equally')
    print('  - Can work well when spatial context is uniform (e.g., MNIST digits)')
    print()
    print('In practice on MNIST, both strategies achieve similar accuracy.')
    print('CLS token is preferred in large-scale settings because it allows the')
    print('model to learn which patches to attend to for the global representation.\n')


# Main function: builds CLS transformer, trains, plots curves, saves model
def main(argv):
    print('Loading data...')
    train_loader, test_loader = get_data(batch_size=64)

    model = NetTransformerCLS(
        image_size=28,
        patch_size=7,
        num_classes=10,
        d_model=64,
        nhead=4,
        num_layers=4,       # deeper than baseline (was 2)
        mlp_dim=128,
        dropout=0.1
    )

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params:,}')
    print()

    print('Training Transformer with CLS token for 5 epochs...')
    train_losses, test_losses, train_accs, test_accs = train_transformer_cls(
        model, train_loader, test_loader, epochs=5)

    plot_cls_curves(train_losses, test_losses, train_accs, test_accs)

    torch.save(model.state_dict(), 'ext3_transformer_cls.pth')
    print('Model saved to ext3_transformer_cls.pth')

    print_comparison_note(test_accs)
    return


if __name__ == '__main__':
    main(sys.argv)
