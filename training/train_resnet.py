"""Training script for the ResNet18 transfer-learning agent.

Usage:
    python -m training.train_resnet --data-path data/raw
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.settings import BATCH_SIZE, CHECKPOINTS_DIR, LEARNING_RATE
from data.dataset import DrivingDataset
from data.preprocessing import preprocess_resnet
from models.resnet_head import ResNetHead
from training.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet18 transfer-learning agent for self-driving"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data directory containing driving_log.csv and IMG/",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=10,
        help="Number of epochs for phase 1 (frozen backbone)",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=20,
        help="Number of epochs for phase 2 (fine-tuning)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Base learning rate (phase 2 uses lr/10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the best checkpoint",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    csv_path = data_path / "driving_log.csv"
    image_root = data_path / "IMG"
    save_path = args.save_path or str(CHECKPOINTS_DIR / "resnet_agent.pth")

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Datasets
    train_ds = DrivingDataset(
        csv_path, image_root, split="train", preprocess_fn=preprocess_resnet
    )
    val_ds = DrivingDataset(
        csv_path, image_root, split="val", preprocess_fn=preprocess_resnet
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = ResNetHead().to(device)

    # ------------------------------------------------------------------
    # Phase 1: Train regression head only (backbone frozen)
    # ------------------------------------------------------------------
    print("\n=== Phase 1: Training head (backbone frozen) ===")
    model.freeze_backbone()

    history_p1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.phase1_epochs,
        learning_rate=args.learning_rate,
        save_path=save_path,
    )

    # ------------------------------------------------------------------
    # Phase 2: Fine-tune last two residual blocks + head
    # ------------------------------------------------------------------
    print("\n=== Phase 2: Fine-tuning (layer3 + layer4 unfrozen) ===")
    model.unfreeze_backbone()

    history_p2 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.phase2_epochs,
        learning_rate=args.learning_rate / 10,
        save_path=save_path,
    )

    # Summary
    best_p1 = min(history_p1["val_loss"])
    best_p2 = min(history_p2["val_loss"])
    print(f"\nBest Phase 1 val loss: {best_p1:.6f}")
    print(f"Best Phase 2 val loss: {best_p2:.6f}")
    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
