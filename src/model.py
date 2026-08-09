"""
model.py

Defines the CNN architecture used to classify facial expressions from
48x48 grayscale images into 7 emotion classes (FER2013 dataset).

This file only defines the model structure -- it does not load data or
train. Both train.py and inference.py import EmotionCNN from here so
there is a single source of truth for the architecture.
"""

import torch.nn as nn


class EmotionCNN(nn.Module):
    """
    CNN for 7-class facial expression classification on 48x48 grayscale images.

    Architecture:
        - 3 convolutional blocks with increasing filter counts (32 -> 64 -> 128)
        - Each block downsamples the spatial size by half via MaxPool2d
        - BatchNorm after each conv layer to stabilize training
        - Dropout after each block to reduce overfitting
        - A fully connected classifier head maps the flattened features to
          7 output logits (one per emotion class)

    Input:  (batch_size, 1, 48, 48)  -- 1 channel because images are grayscale
    Output: (batch_size, 7)          -- raw logits, one per class
            (softmax is applied later, either inside CrossEntropyLoss during
            training, or manually during inference)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Block 1: 1 -> 32 channels, 48x48 -> 24x24
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        # Block 2: 32 -> 64 channels, 24x24 -> 12x12
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        # Block 3: 64 -> 128 channels, 12x12 -> 6x6
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        # Classifier head: flatten the 128 x 6 x 6 feature map and map to 7 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: run one dummy batch through the model and confirm
    # the output shape is (batch_size, 7) as expected.
    import torch

    model = EmotionCNN(num_classes=7)
    dummy_input = torch.randn(4, 1, 48, 48)  # batch of 4 fake grayscale images
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())

    print("Output shape:", output.shape)  # expect: torch.Size([4, 7])
    print("Total parameters:", f"{total_params:,}")
