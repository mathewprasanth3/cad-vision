"""
scripts/train.py

Training script for CADVision MVCNN.
Trains the model on rendered CAD images and saves the best checkpoint.

Usage:
    uv run python scripts/train.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(".")
from src.dataset.featurenet_dataset import get_dataloaders
from src.models.mvcnn import MVCNN


RENDERS_DIR = Path("data/renders")
CHECKPOINT_DIR = Path("checkpoints")
NUM_CLASSES = 24
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 10


def get_device():
    if torch.backends.mps.is_available():
        print("using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("using CUDA GPU")
        return torch.device("cuda")
    else:
        print("using CPU")
        return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    print("CADVision - training")
    print(f"epochs:        {EPOCHS}")
    print(f"batch size:    {BATCH_SIZE}")
    print(f"learning rate: {LEARNING_RATE}")

    device = get_device()

    # data
    print("\nloading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        renders_dir=RENDERS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,  # 0 for MPS compatibility
    )

    # model
    print("building model...")
    model = MVCNN(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # checkpointing
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_acc = 0.0
    epochs_no_improve = 0

    print("\nstarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:02d}/{EPOCHS}  "
            f"train loss: {train_loss:.4f}  train acc: {train_acc:.4f}  "
            f"val loss: {val_loss:.4f}  val acc: {val_acc:.4f}  "
            f"lr: {lr:.6f}  time: {elapsed:.1f}s"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
            print(f"  saved best model (val acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nearly stopping — no improvement for {PATIENCE} epochs")
            break

    # final evaluation on test set
    print("\nloading best model for test evaluation...")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"test loss: {test_loss:.4f}")
    print(f"test acc:  {test_acc:.4f}")
    print(f"\nbest val acc: {best_val_acc:.4f}")
    print(f"model saved to {CHECKPOINT_DIR / 'best_model.pth'}")
    print("next: uv run python scripts/infer.py")


if __name__ == "__main__":
    main()