from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.settings import (
    CAMERA_STEERING_OFFSET,
    STEERING_OVERSAMPLE_THRESHOLD,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from data.augmentation import augment


class DrivingDataset(Dataset):
    """PyTorch Dataset for the Udacity self-driving car behavioral cloning data.

    Loads images and targets (steering, throttle) from CSV metadata.
    Supports left/right camera augmentation with steering offset.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        split: str = "train",
        preprocess_fn=None,
        use_side_cameras: bool = True,
        do_augment: bool = True,
        oversample: bool = True,
    ):
        self.image_root = Path(image_root)
        self.preprocess_fn = preprocess_fn
        self.do_augment = do_augment and (split == "train")
        self.use_side_cameras = use_side_cameras and (split == "train")

        df = pd.read_csv(csv_path)
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Split by sequential blocks to avoid temporal leakage
        n = len(df)
        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        if split == "train":
            df = df.iloc[:train_end]
        elif split == "val":
            df = df.iloc[train_end:val_end]
        elif split == "test":
            df = df.iloc[val_end:]

        # Build samples list: (image_path, steering, throttle)
        self.samples: list[tuple[str, float, float]] = []
        for _, row in df.iterrows():
            steering = float(row["steering_angle"])
            throttle = float(row["throttle"])

            # Center camera
            center_path = str(row["centercam"]).strip()
            self.samples.append((center_path, steering, throttle))

            # Side cameras with offset
            if self.use_side_cameras:
                left_path = str(row["leftcam"]).strip()
                right_path = str(row["rightcam"]).strip()
                self.samples.append(
                    (left_path, steering + CAMERA_STEERING_OFFSET, throttle)
                )
                self.samples.append(
                    (right_path, steering - CAMERA_STEERING_OFFSET, throttle)
                )

        # Oversample frames with significant steering
        if oversample and split == "train":
            extra = [
                s
                for s in self.samples
                if abs(s[1]) > STEERING_OVERSAMPLE_THRESHOLD
            ]
            self.samples.extend(extra)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, steering, throttle = self.samples[idx]

        # Resolve image path (handle relative paths from CSV)
        full_path = self.image_root / Path(img_path).name
        if not full_path.exists():
            # Try using the path as-is
            full_path = Path(img_path)

        image = cv2.imread(str(full_path))
        if image is None:
            # Return a black image as fallback
            image = np.zeros((160, 320, 3), dtype=np.uint8)

        # Augmentation (only during training)
        if self.do_augment:
            image, steering = augment(image, steering)

        # Preprocessing (agent-specific)
        if self.preprocess_fn is not None:
            image = self.preprocess_fn(image)

        # Convert to tensor: HWC -> CHW
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        target_tensor = torch.tensor([steering, throttle], dtype=torch.float32)

        return image_tensor, target_tensor
