import numpy as np


def softmax_with_temperature(
    confidences: list[float], temperature: float = 1.0
) -> np.ndarray:
    """Apply softmax with temperature to normalize confidence scores."""
    c = np.array(confidences, dtype=np.float64)
    c = c / max(temperature, 1e-8)
    exp_c = np.exp(c - np.max(c))  # numerical stability
    return exp_c / exp_c.sum()


def dampen_by_performance(
    confidences: list[float],
    recent_mse: list[float],
    decay_factor: float = 0.5,
) -> list[float]:
    """Reduce confidence for agents with poor recent performance.

    Agents with MSE above the median get their confidence multiplied
    by decay_factor.
    """
    if not recent_mse or all(m == 0 for m in recent_mse):
        return confidences

    median_mse = float(np.median(recent_mse))
    dampened = []
    for conf, mse in zip(confidences, recent_mse):
        if mse > median_mse:
            dampened.append(conf * decay_factor)
        else:
            dampened.append(conf)
    return dampened
