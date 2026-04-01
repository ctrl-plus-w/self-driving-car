from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from agents.base_agent import BaseAgent, Prediction
from config.settings import BATCH_SIZE, CHECKPOINTS_DIR, LEARNING_RATE, NUM_EPOCHS
from data.dataset import DrivingDataset
from data.preprocessing import preprocess_resnet
from models.resnet_head import ResNetHead
from training.trainer import train_model


class ResNetAgent(BaseAgent):
    """Agent 2: ResNet18 transfer-learning agent with Monte Carlo dropout."""

    name: str = "resnet"

    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = ResNetHead().to(self.device)

    # ------------------------------------------------------------------
    # Prediction with Monte Carlo dropout for confidence estimation
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        speed: float,
        history: list[np.ndarray] | None = None,
    ) -> Prediction:
        """Run 5 stochastic forward passes (MC dropout) and return the mean
        prediction with a variance-based confidence score."""

        processed = preprocess_resnet(image)
        tensor = (
            torch.from_numpy(processed.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        # Enable dropout at inference time for Monte Carlo sampling
        self.model.train()

        mc_passes = 5
        predictions = []
        with torch.no_grad():
            for _ in range(mc_passes):
                pred = self.model(tensor)
                predictions.append(pred.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)  # (mc_passes, 2)
        mean_pred = predictions.mean(axis=0)
        variance = predictions.var(axis=0).mean()

        # Confidence: high when variance is low
        confidence = float(1.0 / (1.0 + variance))

        return Prediction(
            steering=float(np.clip(mean_pred[0], -1.0, 1.0)),
            throttle=float(np.clip(mean_pred[1], 0.0, 1.0)),
            confidence=confidence,
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Two-phase training
    # ------------------------------------------------------------------

    def train(self, data_path: str, **kwargs) -> dict:
        """Two-phase training: frozen backbone then fine-tuning.

        Phase 1 — head only: backbone frozen, higher learning rate.
        Phase 2 — fine-tune:  last two blocks unfrozen, lower learning rate.
        """
        data_path = Path(data_path)
        csv_path = data_path / "driving_log.csv"
        image_root = data_path / "IMG"

        # Datasets
        train_ds = DrivingDataset(
            csv_path, image_root, split="train", preprocess_fn=preprocess_resnet
        )
        val_ds = DrivingDataset(
            csv_path, image_root, split="val", preprocess_fn=preprocess_resnet
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        phase1_epochs = kwargs.get("phase1_epochs", 10)
        phase2_epochs = kwargs.get("phase2_epochs", 20)
        lr = kwargs.get("learning_rate", LEARNING_RATE)
        save_path = str(
            kwargs.get("save_path", CHECKPOINTS_DIR / "resnet_agent.pth")
        )

        # Phase 1: train head only with frozen backbone
        print("=== Phase 1: Training head (backbone frozen) ===")
        self.model.freeze_backbone()
        self.model.to(self.device)

        history_p1 = train_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            num_epochs=phase1_epochs,
            learning_rate=lr,
            save_path=save_path,
        )

        # Phase 2: fine-tune last two blocks with lower lr
        print("=== Phase 2: Fine-tuning (layer3 + layer4 unfrozen) ===")
        self.model.unfreeze_backbone()

        history_p2 = train_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            num_epochs=phase2_epochs,
            learning_rate=lr / 10,
            save_path=save_path,
        )

        return {
            "phase1": history_p1,
            "phase2": history_p2,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file."""
        state_dict = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def save(self, checkpoint_path: str) -> None:
        """Save model weights to a checkpoint file."""
        torch.save(self.model.state_dict(), checkpoint_path)
