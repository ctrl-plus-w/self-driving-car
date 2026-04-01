import numpy as np


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predictions - targets)))


def steering_smoothness(steerings: np.ndarray) -> float:
    """Measure smoothness as mean absolute difference between consecutive predictions.

    Lower values mean smoother steering. Returns 0 for single-element arrays.
    """
    if len(steerings) < 2:
        return 0.0
    diffs = np.abs(np.diff(steerings))
    return float(np.mean(diffs))


def compute_all_metrics(
    pred_steering: np.ndarray,
    pred_throttle: np.ndarray,
    true_steering: np.ndarray,
    true_throttle: np.ndarray,
) -> dict[str, float]:
    """Compute a full set of metrics for steering and throttle predictions."""
    return {
        "steering_mse": mse(pred_steering, true_steering),
        "steering_mae": mae(pred_steering, true_steering),
        "throttle_mse": mse(pred_throttle, true_throttle),
        "throttle_mae": mae(pred_throttle, true_throttle),
        "steering_smoothness": steering_smoothness(pred_steering),
    }
