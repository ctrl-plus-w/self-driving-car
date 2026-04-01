"""Training script for the PilotNet agent."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.settings import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from data.dataset import DrivingDataset
from data.preprocessing import preprocess_pilotnet
from models.pilotnet import PilotNet
from training.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PilotNet model")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the data directory containing driving_log.csv and IMG/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "pilotnet_best.pth"),
        help="Path to save the best checkpoint",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    csv_path = data_path / "driving_log.csv"
    image_root = data_path / "IMG"

    # ---- Device auto-detection ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Datasets ----
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
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # ---- Model ----
    model = PilotNet().to(device)
    print(model)

    # ---- Training ----
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.checkpoint,
    )

    # ---- Save final checkpoint (regardless of best val) ----
    final_path = str(Path(args.checkpoint).with_name("pilotnet_final.pth"))
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint to {final_path}")
    print(
        f"Best val loss: {min(history['val_loss']):.6f} "
        f"(epoch {history['val_loss'].index(min(history['val_loss'])) + 1})"
    )


if __name__ == "__main__":
    main()
