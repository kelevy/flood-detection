"""
Evaluation script for the trained flood detection U-Net.

Loads a trained checkpoint, runs inference on the validation split,
and reports segmentation metrics: IoU, F1, precision, recall.

Usage:
    python evaluate.py --checkpoint ../models/best_model.pt
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split

from dataset import Sen1Floods11Dataset
from model import build_model


S1_DIR = "../data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand"
LABEL_DIR = "../data/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"
VAL_SPLIT = 0.2
SEED = 42  # must match train.py / train_vertex.py so we evaluate on the SAME val split


def get_val_dataset():
    full_dataset = Sen1Floods11Dataset(S1_DIR, LABEL_DIR)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    return val_ds


def compute_metrics(preds, labels, ignore_index=-1):
    """
    Compute per-class and averaged IoU, precision, recall, F1 over valid pixels.
    """
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]

    results = {}
    for cls, name in [(0, "not_water"), (1, "water")]:
        pred_cls = preds == cls
        label_cls = labels == cls

        tp = (pred_cls & label_cls).sum().item()
        fp = (pred_cls & ~label_cls).sum().item()
        fn = (~pred_cls & label_cls).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    results["mean_iou"] = (results["not_water"]["iou"] + results["water"]["iou"]) / 2
    return results


def load_trained_model(checkpoint_path, device):
    """Convenience function for loading a trained model."""
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../models/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="../results")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU evaluation for reproducible results (MPS has known non-determinism)")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print(f"Using device: {device}")

    model = load_trained_model(args.checkpoint, device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    val_ds = get_val_dataset()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(val_ds)} validation samples")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            all_preds.append(preds.flatten())
            all_labels.append(labels.flatten())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_preds, all_labels)

    print("\n=== Evaluation Results ===")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("\nWater class:")
    print(f"  IoU:       {metrics['water']['iou']:.4f}")
    print(f"  Precision: {metrics['water']['precision']:.4f}")
    print(f"  Recall:    {metrics['water']['recall']:.4f}")
    print(f"  F1:        {metrics['water']['f1']:.4f}")
    print("\nNot-water class:")
    print(f"  IoU:       {metrics['not_water']['iou']:.4f}")
    print(f"  Precision: {metrics['not_water']['precision']:.4f}")
    print(f"  Recall:    {metrics['not_water']['recall']:.4f}")
    print(f"  F1:        {metrics['not_water']['f1']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "metrics.txt")
    with open(results_path, "w") as f:
        f.write("Flood Detection Model Evaluation\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n\n")
        for cls_name in ["water", "not_water"]:
            f.write(f"{cls_name}:\n")
            for metric_name, value in metrics[cls_name].items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")
    print(f"\nMetrics saved to {results_path}")


if __name__ == "__main__":
    main()