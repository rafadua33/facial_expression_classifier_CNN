"""
train.py

Trains the EmotionCNN model on the FER2013 dataset and saves the
best-performing checkpoint (by validation accuracy) to disk.

Usage:
    python train.py

Run this in an environment with a GPU (e.g. Colab) for reasonable
training times -- on CPU this will be very slow.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

from dataset import get_dataloaders
from model import EmotionCNN


# ---- Config ----
DATA_DIR = "fer2013_data"
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
CHECKPOINT_PATH = "best.pt"


def get_class_weights(train_loader, num_classes, device):
    """
    Computes per-class weights to counteract FER2013's class imbalance
    (the 'disgust' class has far fewer examples than the others).
    Underrepresented classes get a higher weight so the loss function
    penalizes mistakes on them more heavily, discouraging the model
    from just ignoring rare classes.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    unique_labels = np.unique(all_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=all_labels,
    )
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full pass over the training set, updating model weights."""
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Runs one full pass over a dataset WITHOUT updating weights.
    Used for both validation (during training) and final test evaluation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def plot_history(history):
    """Plots train/val loss and accuracy curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Data ----
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    num_classes = len(class_names)

    # ---- Model ----
    model = EmotionCNN(num_classes=num_classes).to(device)

    # ---- Loss, optimizer, scheduler ----
    class_weights = get_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # ---- Training loop ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    plot_history(history)

    # ---- Final test evaluation using the best saved checkpoint ----
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
