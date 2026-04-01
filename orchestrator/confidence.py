from __future__ import annotations

import numpy as np

from config.settings import OUTLIER_PENALTY


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
    decay_alpha: float = 5.0,
) -> list[float]:
    """Reduce confidence for agents with poor recent performance.

    Uses continuous exponential decay: conf * exp(-alpha * mse).
    Agents with MSE=0 keep full confidence, higher MSE -> lower confidence.
    """
    if not recent_mse or all(m == 0 for m in recent_mse):
        return list(confidences)

    dampened = []
    for conf, mse in zip(confidences, recent_mse):
        dampened.append(conf * float(np.exp(-decay_alpha * mse)))
    return dampened


def detect_outliers_mad(
    values: list[float],
    threshold: float = 2.0,
) -> list[float]:
    """Detect outliers using Median Absolute Deviation (MAD).

    Returns a list of penalty multipliers (1.0 = normal, OUTLIER_PENALTY = outlier).
    Uses modified Z-scores: 0.6745 * |x - median| / MAD.
    """
    if len(values) < 2:
        return [1.0] * len(values)

    arr = np.array(values, dtype=np.float64)
    median = np.median(arr)
    abs_deviations = np.abs(arr - median)
    mad = np.median(abs_deviations)

    if mad < 1e-8:
        # All agents nearly agree — flag any with non-trivial deviation
        penalties = []
        for dev in abs_deviations:
            if dev > 0.05:  # small epsilon for floating-point tolerance
                penalties.append(OUTLIER_PENALTY)
            else:
                penalties.append(1.0)
        return penalties

    modified_z = 0.6745 * abs_deviations / mad
    return [
        OUTLIER_PENALTY if z > threshold else 1.0
        for z in modified_z
    ]


def compute_agreement_penalties(
    steerings: list[float],
    throttles: list[float],
    threshold: float = 2.0,
) -> list[float]:
    """Compute per-agent penalty multipliers based on inter-agent agreement.

    An agent that is an outlier on *either* steering or throttle gets penalized.
    """
    steering_penalties = detect_outliers_mad(steerings, threshold)
    throttle_penalties = detect_outliers_mad(throttles, threshold)

    # Take the minimum (most severe) penalty across dimensions
    return [
        min(sp, tp) for sp, tp in zip(steering_penalties, throttle_penalties)
    ]
