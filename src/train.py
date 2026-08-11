"""
Training script for flood segmentation U-Net on Sen1Floods11.

Handles:
    - train/val split
    - masking out "no data" pixels (label == -1) from the loss
    - cross-entropy loss
    - IoU metric tracking
    - checkpointing the best model
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import Sen1Floods11Dataset
from model import build_model


# ---- Config ----
S1_DIR = "../data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand"
LABEL_DIR = "../data/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"
CHECKPOINT_DIR = "../models"
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
SEED = 42


def compute_iou(preds, labels, ignore_index=-1, num_classes=2):
    """
    Compute mean IoU over valid pixels (excludes ignore_index).

    Args:
        preds (Tensor): (B, H, W) predicted class indices
        labels (Tensor): (B, H, W) ground truth class indices
    """
    valid = labels != ignore_index
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid
        label_cls = (labels == cls) & valid
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # (B, 2, H, W)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        total_iou += compute_iou(preds, labels) * images.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_iou / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- Data ----
    full_dataset = Sen1Floods11Dataset(S1_DIR, LABEL_DIR)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # ---- Model ----
    model = build_model().to(device)

    # ignore_index=-1 excludes "no data" pixels from the loss entirely
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---- Training loop ----
    best_iou = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (IoU: {best_iou:.4f})")

    print(f"Training complete. Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()