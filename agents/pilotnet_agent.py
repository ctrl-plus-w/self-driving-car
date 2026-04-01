from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent, Prediction
from config.settings import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from data.dataset import DrivingDataset
from data.preprocessing import preprocess_pilotnet
from models.pilotnet import PilotNet
from training.trainer import train_model


class PilotNetAgent(BaseAgent):
    """Agent 1 — NVIDIA PilotNet end-to-end driving model."""

    name: str = "pilotnet"

    def __init__(self):
        # Device auto-detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = PilotNet().to(self.device)
        self.model.eval()

        # Rolling window for confidence estimation
        self._recent_predictions: deque[np.ndarray] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        speed: float,
        history: list[np.ndarray] | None = None,
    ) -> Prediction:
        """Preprocess *image*, run inference, and return a Prediction."""
        processed = preprocess_pilotnet(image)

        # HWC -> CHW tensor, add batch dim
        tensor = (
            torch.from_numpy(processed.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            output = self.model(tensor).cpu().numpy().squeeze()  # shape (2,)

        steering = float(np.clip(output[0], -1.0, 1.0))
        throttle = float(np.clip(output[1], 0.0, 1.0))

        # Confidence: inverse of recent prediction variance
        self._recent_predictions.append(output)
        if len(self._recent_predictions) >= 2:
            stacked = np.stack(list(self._recent_predictions))
            variance = float(stacked.var())
            # Map variance to confidence in [0, 1] — low variance ⇒ high confidence
            confidence = float(np.clip(1.0 / (1.0 + variance * 100.0), 0.0, 1.0))
        else:
            confidence = 0.5  # not enough history yet

        return Prediction(
            steering=steering,
            throttle=throttle,
            confidence=confidence,
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data_path: str, **kwargs) -> dict:
        """Create datasets, data-loaders, and train the PilotNet model."""
        data_path = Path(data_path)
        csv_path = data_path / "driving_log.csv"
        image_root = data_path / "IMG"

        train_ds = DrivingDataset(
            csv_path=csv_path,
            image_root=image_root,
            split="train",
            preprocess_fn=preprocess_pilotnet,
        )
        val_ds = DrivingDataset(
            csv_path=csv_path,
            image_root=image_root,
            split="val",
            preprocess_fn=preprocess_pilotnet,
            do_augment=False,
        )

        batch_size = kwargs.get("batch_size", BATCH_SIZE)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        num_epochs = kwargs.get("num_epochs", NUM_EPOCHS)
        lr = kwargs.get("learning_rate", LEARNING_RATE)
        save_path = kwargs.get("save_path", None)

        self.model.train()
        history = train_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            num_epochs=num_epochs,
            learning_rate=lr,
            save_path=save_path,
        )
        self.model.eval()
        return history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file."""
        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def save(self, checkpoint_path: str) -> None:
        """Save current model weights to disk."""
        torch.save(self.model.state_dict(), checkpoint_path)
