"""
Training script for greenhouse crop ripeness classification.

Fine-tunes a pretrained MobileNetV2 on a folder-structured image dataset:

    data/
        train/
            ripe/
            unripe/
            rotten/
        val/
            ripe/
            unripe/
            rotten/

Any dataset that follows this ImageFolder layout works (e.g. a Kaggle
fruit/vegetable ripeness dataset, renamed into these class folders).

Usage:
    python train_model.py --data_dir ./data --epochs 10 --output ../app/model/ripeness_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze the pretrained feature extractor; only train the new classifier head.
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_dataloaders(data_dir: str, batch_size: int):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(str(Path(data_dir) / "val"), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.classes


@torch.no_grad()
def evaluate(model, val_loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total if total else 0.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Classes: {class_names}")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    best_val_acc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, output_path)
            print(f"  -> Saved new best model (val_acc={val_acc:.4f}) to {output_path}")

    class_names_path = output_path.with_name("class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"Class names saved to: {class_names_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train greenhouse ripeness classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset root containing train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="../app/model/ripeness_model.pt",
                         help="Where to save the trained model checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
