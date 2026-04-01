import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.settings import STEERING_LOSS_WEIGHT


def weighted_mse_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """MSE loss with higher weight on steering (column 0) vs throttle (column 1)."""
    weights = torch.tensor(
        [STEERING_LOSS_WEIGHT, 1.0], device=pred.device
    )
    return (weights * (pred - target) ** 2).mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = weighted_mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Validate model. Returns average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss = weighted_mse_loss(predictions, targets)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    save_path: str | None = None,
) -> dict:
    """Full training loop with validation and optional checkpoint saving."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} — "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")

    return history
