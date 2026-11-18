"""
Deep Learning Classifier Training Script for Macadamia Defect Grading.
Implements transfer learning (MobileNetV2 / ResNet18), weighted Cross-Entropy Loss
for handling class imbalance, data augmentation, evaluation, and ONNX export.
"""

import argparse
import os
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_defect_classifier(
    data_dir: str,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_onnx: str = "src/models/weights/defect_classifier.onnx",
):
    """
    Trains a deep learning classifier on cropped macadamia kernels.
    Classes:
    0: Healthy / Premium
    1: Cracked / Broken
    2: Discolored / Brown Center
    3: Moldy / Insect Damaged
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms, models
    except ImportError:
        logger.warning(
            "PyTorch or Torchvision is not installed in the current environment.\n"
            "This script is provided for GPU fine-tuning environments.\n"
            "Pre-compiled ONNX models and the optical feature inference engine are ready in `src/models/weights/`."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device for training: %s", device)

    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir):
        logger.error("Training data directory not found at: %s", train_dir)
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model Backbone: MobileNetV2 with custom 4-class classification head
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )
    model = model.to(device)

    # Class-weighted loss to handle imbalance (defects are rarer than healthy nuts)
    # Higher weights on moldy/insect and cracked
    class_weights = torch.tensor([1.0, 2.5, 2.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Starting classifier training loop...")
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss/total:.4f} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            # Export to ONNX
            os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
            dummy_input = torch.randn(1, 3, 128, 128, device=device)
            torch.onnx.export(
                model,
                dummy_input,
                output_onnx,
                input_names=["input"],
                output_names=["probabilities"],
                opset_version=14,
            )
            logger.info("Saved best model to ONNX: %s (Val Acc: %.3f)", output_onnx, best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Macadamia Defect Classifier")
    parser.add_argument("--data_dir", type=str, default="data/processed/crops", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_onnx", type=str, default="src/models/weights/defect_classifier.onnx")
    args = parser.parse_args()

    train_defect_classifier(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_onnx=args.output_onnx,
    )
