"""
PyTorch Dataset for Sen1Floods11 flood detection.

Loads Sentinel-1 SAR image chips (VV, VH bands) and their corresponding
hand-labeled flood masks.

Label values:
    -1 : No data / not valid (masked out during training)
     0 : Not water
     1 : Water
"""

import os
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, random_split


class Sen1Floods11Dataset(Dataset):
    def __init__(self, s1_dir, label_dir, transform=None):
        """
        Args:
            s1_dir (str): path to folder containing S1Hand .tif files
            label_dir (str): path to folder containing LabelHand .tif files
            transform: optional albumentations transform applied to both
                       image and mask together
        """
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.transform = transform

        # Build list of chip IDs by matching S1 files to their labels
        s1_files = sorted(glob.glob(os.path.join(s1_dir, "*_S1Hand.tif")))
        self.samples = []
        for s1_path in s1_files:
            basename = os.path.basename(s1_path)
            chip_id = basename.replace("_S1Hand.tif", "")
            label_path = os.path.join(label_dir, f"{chip_id}_LabelHand.tif")
            if os.path.exists(label_path):
                self.samples.append((s1_path, label_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matching S1/label pairs found in {s1_dir} and {label_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_s1(self, path):
        """Load a 2-band SAR image (VV, VH) and normalize."""
        with rasterio.open(path) as src:
            img = src.read()  # shape: (2, H, W), dtype float32, unit dB

        # Replace any NaN/inf from missing data
        img = np.nan_to_num(img, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)

        # Clip to a sane dB range for SAR backscatter, then normalize to [0, 1]
        img = np.clip(img, -50, 1)
        img = (img + 50) / 51.0

        return img.astype(np.float32)

    def _load_label(self, path):
        """Load the flood label mask."""
        with rasterio.open(path) as src:
            label = src.read(1)  # shape: (H, W)
        return label.astype(np.int64)

    def __getitem__(self, idx):
        s1_path, label_path = self.samples[idx]

        image = self._load_s1(s1_path)      # (2, H, W)
        label = self._load_label(label_path)  # (H, W)

        if self.transform:
            # albumentations expects HWC image and HW mask
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=label)
            image = np.transpose(augmented["image"], (2, 0, 1))
            label = augmented["mask"]

        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).long()

        return image_tensor, label_tensor


def get_splits(full_dataset, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Split a dataset into train / validation / test subsets.

    - train: used to update model weights
    - val: used during training to select the best checkpoint (never
           backpropagated on, but does influence which checkpoint is kept)
    - test: held out completely; touched only once, after all training
            and model selection is finished, to report final metrics

    Args:
        full_dataset: the full Sen1Floods11Dataset
        train_frac (float): fraction for training (default 0.8)
        val_frac (float): fraction for validation (default 0.1)
                          remainder (default 0.1) goes to test
        seed (int): random seed for reproducible splits

    Returns:
        (train_ds, val_ds, test_ds)
    """
    n = len(full_dataset)
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = n - train_size - val_size  # remainder, avoids rounding gaps

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Sanity check
    s1_dir = "data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand"
    label_dir = "data/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"

    dataset = Sen1Floods11Dataset(s1_dir, label_dir)
    print(f"Found {len(dataset)} chip pairs")

    image, label = dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Label shape: {label.shape}, unique values: {torch.unique(label)}")

    train_ds, val_ds, test_ds = get_splits(dataset)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")