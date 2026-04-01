from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents.base_agent import BaseAgent, Prediction
from config.settings import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    SAFE_THROTTLE,
    SEQUENCE_LENGTH,
)
from data.preprocessing import preprocess_pilotnet
from data.sequence_dataset import SequenceDataset
from models.cnn_lstm import CNNLSTM
from training.trainer import weighted_mse_loss


class TemporalAgent(BaseAgent):
    """Agent 4 — CNN-LSTM temporal model that reasons over a sequence of frames."""

    name: str = "temporal"

    def __init__(self):
        # Device auto-detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = CNNLSTM().to(self.device)
        self.model.eval()

        # Internal frame buffer — stores preprocessed CHW tensors.
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

        # Hidden-state norm history for confidence estimation.
        self._hidden_norms: deque[float] = deque(maxlen=5)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        speed: float,
        history: list[np.ndarray] | None = None,
    ) -> Prediction:
        """Preprocess *image*, append to buffer, and run temporal inference.

        Until the buffer is full (SEQUENCE_LENGTH frames), a safe default
        prediction is returned.
        """
        # Preprocess and store HWC -> CHW
        processed = preprocess_pilotnet(image)
        self._frame_buffer.append(processed.transpose(2, 0, 1))

        if len(self._frame_buffer) < SEQUENCE_LENGTH:
            return Prediction(
                steering=0.0,
                throttle=SAFE_THROTTLE,
                confidence=0.0,
                agent_name=self.name,
            )

        # Build input tensors from the buffer.
        images_np = np.stack(list(self._frame_buffer))  # (seq, C, H, W)
        images_tensor = (
            torch.from_numpy(images_np).float().unsqueeze(0).to(self.device)
        )  # (1, seq, C, H, W)

        # Use the provided speed for every frame (we only have the current one).
        speeds_tensor = (
            torch.full((1, SEQUENCE_LENGTH), speed, dtype=torch.float32)
            .to(self.device)
        )

        with torch.no_grad():
            # Run model and capture hidden state for confidence estimation.
            output = self.model(images_tensor, speeds_tensor)  # (1, 2)

            # Get the last LSTM hidden state norm for confidence tracking.
            # Re-run through CNN + LSTM internals to grab h_n.
            batch_size, seq_len = images_tensor.shape[:2]
            imgs_flat = images_tensor.view(
                batch_size * seq_len, *images_tensor.shape[2:]
            )
            features = self.model.cnn(imgs_flat).view(batch_size, seq_len, -1)
            speeds_expanded = speeds_tensor.unsqueeze(-1)
            lstm_input = torch.cat([features, speeds_expanded], dim=-1)
            _, (h_n, _) = self.model.lstm(lstm_input)
            hidden_norm = float(h_n[-1].norm().cpu())

        self._hidden_norms.append(hidden_norm)

        pred = output.cpu().numpy().squeeze()  # (2,)
        steering = float(np.clip(pred[0], -1.0, 1.0))
        throttle = float(np.clip(pred[1], 0.0, 1.0))

        # Confidence: based on stability of hidden-state norms.
        if len(self._hidden_norms) >= 2:
            norms = np.array(self._hidden_norms)
            variance = float(norms.var())
            confidence = float(np.clip(1.0 / (1.0 + variance * 10.0), 0.0, 1.0))
        else:
            confidence = 0.5

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
        """Create SequenceDataset loaders and run a custom training loop."""
        data_path = Path(data_path)
        csv_path = data_path / "driving_log.csv"
        image_root = data_path / "IMG"

        train_ds = SequenceDataset(
            csv_path=csv_path, image_root=image_root, split="train"
        )
        val_ds = SequenceDataset(
            csv_path=csv_path, image_root=image_root, split="val"
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            # --- Train ---
            self.model.train()
            total_train_loss = 0.0
            for images, speeds, targets in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
            ):
                images = images.to(self.device)
                speeds = speeds.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(images, speeds)
                loss = weighted_mse_loss(predictions, targets)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)

            train_loss = total_train_loss / len(train_loader.dataset)

            # --- Validate ---
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for images, speeds, targets in tqdm(
                    val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
                ):
                    images = images.to(self.device)
                    speeds = speeds.to(self.device)
                    targets = targets.to(self.device)

                    predictions = self.model(images, speeds)
                    loss = weighted_mse_loss(predictions, targets)
                    total_val_loss += loss.item() * images.size(0)

            val_loss = total_val_loss / len(val_loader.dataset)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs} — "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model to {save_path}")

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
