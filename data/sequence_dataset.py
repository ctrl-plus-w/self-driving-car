from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.settings import SEQUENCE_LENGTH, TRAIN_SPLIT, VAL_SPLIT
from data.preprocessing import preprocess_pilotnet


class SequenceDataset(Dataset):
    """Dataset that yields sequences of consecutive frames for the temporal (LSTM) agent.

    Each item is a tuple of:
      - images: Tensor of shape (seq_len, C, H, W)
      - speeds: Tensor of shape (seq_len,)
      - target: Tensor of shape (2,) — (steering, throttle) for the last frame
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        split: str = "train",
        seq_len: int = SEQUENCE_LENGTH,
    ):
        self.image_root = Path(image_root)
        self.seq_len = seq_len

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        n = len(df)
        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        if split == "train":
            df = df.iloc[:train_end]
        elif split == "val":
            df = df.iloc[train_end:val_end]
        elif split == "test":
            df = df.iloc[val_end:]

        df = df.reset_index(drop=True)

        self.image_paths = [str(p).strip() for p in df["centercam"]]
        self.steerings = df["steering_angle"].values.astype(np.float32)
        self.throttles = df["throttle"].values.astype(np.float32)
        self.speeds = df["speed"].values.astype(np.float32)

        # Normalize speed to [0, 1]
        max_speed = self.speeds.max() if self.speeds.max() > 0 else 1.0
        self.speeds = self.speeds / max_speed

    def __len__(self) -> int:
        return max(0, len(self.image_paths) - self.seq_len + 1)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = []
        for i in range(idx, idx + self.seq_len):
            full_path = self.image_root / Path(self.image_paths[i]).name
            if not full_path.exists():
                full_path = Path(self.image_paths[i])

            image = cv2.imread(str(full_path))
            if image is None:
                image = np.zeros((160, 320, 3), dtype=np.uint8)

            image = preprocess_pilotnet(image)
            # HWC -> CHW
            images.append(image.transpose(2, 0, 1))

        last = idx + self.seq_len - 1
        images_tensor = torch.from_numpy(np.stack(images)).float()
        speeds_tensor = torch.from_numpy(self.speeds[idx : idx + self.seq_len])
        target_tensor = torch.tensor(
            [self.steerings[last], self.throttles[last]], dtype=torch.float32
        )

        return images_tensor, speeds_tensor, target_tensor
