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
from torch.utils.data import Dataset


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


if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify the dataset loads
    s1_dir = "data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand"
    label_dir = "data/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand"

    dataset = Sen1Floods11Dataset(s1_dir, label_dir)
    print(f"Found {len(dataset)} chip pairs")

    image, label = dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Label shape: {label.shape}, unique values: {torch.unique(label)}")