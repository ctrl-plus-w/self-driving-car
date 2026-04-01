"""Training script for the CNN-LSTM temporal agent (Agent 4).

Usage:
    python -m training.train_temporal --csv data/raw/driving_log.csv --images data/raw/IMG
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.settings import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from data.sequence_dataset import SequenceDataset
from models.cnn_lstm import CNNLSTM
from training.trainer import weighted_mse_loss


def train_temporal(
    csv_path: str,
    image_root: str,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> dict:
    """Run the full training loop for the CNN-LSTM model."""

    # Device auto-detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = SequenceDataset(csv_path=csv_path, image_root=image_root, split="train")
    val_ds = SequenceDataset(csv_path=csv_path, image_root=image_root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    # Model
    model = CNNLSTM().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    save_path = CHECKPOINTS_DIR / "cnn_lstm_best.pth"
    best_val_loss = float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        for images, speeds, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            images = images.to(device)
            speeds = speeds.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images, speeds)
            loss = weighted_mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, speeds, targets in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
            ):
                images = images.to(device)
                speeds = speeds.to(device)
                targets = targets.to(device)

                predictions = model(images, speeds)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the CNN-LSTM temporal model (Agent 4)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to driving_log.csv",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to root directory containing images (e.g. data/raw/IMG)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)

    args = parser.parse_args()

    train_temporal(
        csv_path=args.csv,
        image_root=args.images,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
