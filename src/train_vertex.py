"""
Training script for flood segmentation U-Net on Sen1Floods11.
Designed to run on Vertex AI, reading data from Google Cloud Storage
and writing checkpoints back to GCS.

Handles:
    - downloading data from GCS at job start
    - train/val split
    - masking out "no data" pixels (label == -1) from the loss
    - cross-entropy loss
    - IoU metric tracking
    - checkpointing the best model, uploaded to GCS
"""

import os
import argparse
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import Sen1Floods11Dataset
from src.model import build_model


def download_data_from_gcs(bucket_name, local_dir="/tmp/data"):
    """Download S1Hand and LabelHand folders from GCS to local disk."""
    os.makedirs(local_dir, exist_ok=True)

    s1_local = os.path.join(local_dir, "S1Hand")
    label_local = os.path.join(local_dir, "LabelHand")

    if not os.path.exists(s1_local):
        print(f"Downloading S1Hand from gs://{bucket_name}/S1Hand ...")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"gs://{bucket_name}/S1Hand", local_dir],
            check=True,
        )

    if not os.path.exists(label_local):
        print(f"Downloading LabelHand from gs://{bucket_name}/LabelHand ...")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"gs://{bucket_name}/LabelHand", local_dir],
            check=True,
        )

    return s1_local, label_local


def upload_checkpoint_to_gcs(local_path, bucket_name, remote_path):
    """Upload a checkpoint file to GCS."""
    subprocess.run(
        ["gsutil", "cp", local_path, f"gs://{bucket_name}/{remote_path}"],
        check=True,
    )


def compute_iou(preds, labels, ignore_index=-1, num_classes=2):
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
        outputs = model(images)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True,
                         help="GCS bucket name containing S1Hand/LabelHand")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    s1_dir, label_dir = download_data_from_gcs(args.bucket)

    full_dataset = Sen1Floods11Dataset(s1_dir, label_dir)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # ---- Model ----
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    best_iou = 0.0
    local_checkpoint = "/tmp/best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), local_checkpoint)
            upload_checkpoint_to_gcs(local_checkpoint, args.bucket, "checkpoints/best_model.pt")
            print(f"  -> New best model saved and uploaded (IoU: {best_iou:.4f})")

    print(f"Training complete. Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()