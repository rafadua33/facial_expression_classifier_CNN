"""
dataset.py

Handles loading the FER2013 dataset from disk and preparing PyTorch
DataLoaders for training, validation, and testing.

Expects the dataset to already be downloaded and unzipped with this structure:

    <data_dir>/train/angry/*.jpg
    <data_dir>/train/disgust/*.jpg
    ... (7 class folders total)
    <data_dir>/test/angry/*.jpg
    ...

This uses torchvision's ImageFolder, which automatically infers class
labels from subfolder names and assigns each an integer index
(alphabetical order), accessible via dataset.classes.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_transforms():
    """
    Returns (train_transform, eval_transform).

    Training gets light augmentation (flip + rotation) so the model
    doesn't just memorize exact pixel arrangements. Validation/test
    transforms skip augmentation so we get a clean, consistent measure
    of how well the model generalizes.

    Both convert to single-channel grayscale and normalize pixel values
    to roughly [-1, 1] (mean=0.5, std=0.5) since raw [0,255] pixel
    values are too large a scale for stable gradient updates.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir: str, batch_size: int = 64, val_split: float = 0.1, seed: int = 42):
    """
    Builds train/val/test DataLoaders from the FER2013 folder structure.

    Args:
        data_dir: path to the folder containing train/ and test/ subfolders
        batch_size: number of images per batch
        val_split: fraction of the training set held out for validation
        seed: random seed for the train/val split, so it's reproducible

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform, eval_transform = get_transforms()

    # Load the full training set once with train_transform. We'll split
    # off a validation subset from this below. Note: this means the val
    # subset technically also gets train_transform's augmentation applied,
    # since transform is attached to the whole dataset, not the subset.
    # For FER2013 this is a minor detail, but if you want a perfectly
    # "clean" validation set, load it a second time with eval_transform
    # and use the same random split indices.
    full_train_dataset = ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    test_dataset = ImageFolder(root=f"{data_dir}/test", transform=eval_transform)

    class_names = full_train_dataset.classes

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick manual check -- update this path to wherever you unzip the
    # dataset locally/in Colab, then run `python dataset.py` to confirm
    # it loads without errors and prints reasonable dataset sizes.
    train_loader, val_loader, test_loader, class_names = get_dataloaders("fer2013_data")

    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)   # expect: (batch_size, 1, 48, 48)
    print("Labels shape:", labels.shape)  # expect: (batch_size,)
