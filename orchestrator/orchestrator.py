from collections import defaultdict

import numpy as np

from agents.base_agent import BaseAgent, Prediction
from config.settings import (
    CONFIDENCE_TEMPERATURE,
    OUTLIER_MAD_THRESHOLD,
    PERFORMANCE_DECAY_ALPHA,
    PERFORMANCE_EMA_GAMMA,
    PERFORMANCE_HISTORY_WINDOW,
    USE_WEIGHTED_MEDIAN,
)
from orchestrator.confidence import (
    compute_agreement_penalties,
    dampen_by_performance,
    softmax_with_temperature,
)
from orchestrator.safety import apply_safety_fallback


class Orchestrator:
    """Robust confidence-weighted ensemble orchestrator.

    Collects predictions from multiple agents, detects outliers via MAD,
    weights by confidence (dampened by recent performance), and blends
    using weighted median with safety clamping.
    """

    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.performance_history: dict[str, list[float]] = defaultdict(list)
        self._ema_mse: dict[str, float] = defaultdict(float)
        self._history_window = PERFORMANCE_HISTORY_WINDOW
        self._last_predictions: list[Prediction] = []

    def predict(
        self,
        image: np.ndarray,
        speed: float,
        frame_history: list[np.ndarray] | None = None,
    ) -> Prediction:
        """Get predictions from all agents and return blended result."""
        predictions: list[Prediction] = []
        for agent in self.agents:
            try:
                predictions.append(agent.predict(image, speed, frame_history))
            except Exception:
                pass  # Skip agents that fail (e.g. untrained)

        if not predictions:
            self._last_predictions = []
            return Prediction(
                steering=0.0, throttle=0.2, confidence=0.0, agent_name="orchestrator"
            )
        return self._blend(predictions)

    def _blend(self, predictions: list[Prediction]) -> Prediction:
        """Outlier-robust confidence-weighted blending."""
        self._last_predictions = predictions
        raw_confidences = [p.confidence for p in predictions]

        # Step 1: Detect and penalize outlier predictions (MAD-based)
        steerings = [p.steering for p in predictions]
        throttles = [p.throttle for p in predictions]
        agreement_penalties = compute_agreement_penalties(
            steerings, throttles, OUTLIER_MAD_THRESHOLD
        )
        penalized = [c * p for c, p in zip(raw_confidences, agreement_penalties)]

        # Step 2: Dampen agents with poor recent performance (exponential decay)
        recent_mse = [
            self._ema_mse[p.agent_name] for p in predictions
        ]
        dampened = dampen_by_performance(penalized, recent_mse, PERFORMANCE_DECAY_ALPHA)

        # Step 3: Normalize via softmax
        weights = softmax_with_temperature(dampened, CONFIDENCE_TEMPERATURE)

        # Step 4: Aggregate — weighted median (robust) or weighted mean
        if USE_WEIGHTED_MEDIAN:
            steering = self._weighted_median(steerings, weights)
            throttle = self._weighted_median(throttles, weights)
        else:
            steering = sum(w * s for w, s in zip(weights, steerings))
            throttle = sum(w * t for w, t in zip(weights, throttles))

        # Step 5: Safety fallback
        max_conf = max(raw_confidences) if raw_confidences else 0.0
        steering, throttle = apply_safety_fallback(steering, throttle, max_conf)

        return Prediction(
            steering=steering,
            throttle=throttle,
            confidence=float(max_conf),
            agent_name="orchestrator",
        )

    @staticmethod
    def _weighted_median(values: list[float], weights: np.ndarray) -> float:
        """Compute weighted median — robust to outliers (50% breakdown point)."""
        pairs = sorted(zip(values, weights), key=lambda x: x[0])
        cumulative = 0.0
        total = float(np.sum(weights))
        half = total / 2.0

        for i, (val, w) in enumerate(pairs):
            cumulative += w
            if cumulative >= half:
                # Interpolate with previous value for smoothness
                if i > 0 and cumulative - w < half:
                    prev_val, prev_w = pairs[i - 1]
                    frac = (half - (cumulative - w)) / w
                    return prev_val + frac * (val - prev_val)
                return val
        return pairs[-1][0]

    def update_performance(
        self, agent_name: str, squared_error: float
    ) -> None:
        """Record a squared error for an agent (used during evaluation)."""
        # Update EMA for responsive dampening
        gamma = PERFORMANCE_EMA_GAMMA
        self._ema_mse[agent_name] = (
            (1 - gamma) * self._ema_mse[agent_name] + gamma * squared_error
        )

        # Keep raw history as well
        history = self.performance_history[agent_name]
        history.append(squared_error)
        if len(history) > self._history_window * 2:
            self.performance_history[agent_name] = history[
                -self._history_window :
            ]
