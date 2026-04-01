import numpy as np

from config.settings import MIN_CONFIDENCE_THRESHOLD, SAFE_THROTTLE


def clamp_prediction(steering: float, throttle: float) -> tuple[float, float]:
    """Clamp steering to [-1, 1] and throttle to [0, 1]."""
    steering = float(np.clip(steering, -1.0, 1.0))
    throttle = float(np.clip(throttle, 0.0, 1.0))
    return steering, throttle


def apply_safety_fallback(
    steering: float,
    throttle: float,
    max_confidence: float,
) -> tuple[float, float]:
    """If all agents have low confidence, fall back to safe defaults.

    Reduces throttle and gently steers toward center (steering=0).
    """
    if max_confidence < MIN_CONFIDENCE_THRESHOLD:
        steering = steering * 0.5  # Dampen steering toward center
        throttle = min(throttle, SAFE_THROTTLE)  # Reduce speed
    return clamp_prediction(steering, throttle)
