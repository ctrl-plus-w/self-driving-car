from collections import defaultdict

import numpy as np

from agents.base_agent import BaseAgent, Prediction
from config.settings import CONFIDENCE_TEMPERATURE
from orchestrator.confidence import dampen_by_performance, softmax_with_temperature
from orchestrator.safety import apply_safety_fallback, clamp_prediction


class Orchestrator:
    """Confidence-weighted ensemble orchestrator.

    Collects predictions from multiple agents, weights them by confidence
    (dampened by recent performance), and blends into a final prediction
    with safety clamping.
    """

    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.performance_history: dict[str, list[float]] = defaultdict(list)
        self._history_window = 50  # track last N errors per agent

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
            return Prediction(
                steering=0.0, throttle=0.2, confidence=0.0, agent_name="orchestrator"
            )
        return self._blend(predictions)

    def _blend(self, predictions: list[Prediction]) -> Prediction:
        """Confidence-weighted blending with performance dampening."""
        raw_confidences = [p.confidence for p in predictions]

        # Dampen agents with poor recent performance
        recent_mse = [
            np.mean(self.performance_history[p.agent_name][-self._history_window:])
            if self.performance_history[p.agent_name]
            else 0.0
            for p in predictions
        ]
        dampened = dampen_by_performance(raw_confidences, recent_mse)

        # Normalize via softmax
        weights = softmax_with_temperature(dampened, CONFIDENCE_TEMPERATURE)

        # Weighted average
        steering = sum(w * p.steering for w, p in zip(weights, predictions))
        throttle = sum(w * p.throttle for w, p in zip(weights, predictions))

        # Safety
        max_conf = max(raw_confidences) if raw_confidences else 0.0
        steering, throttle = apply_safety_fallback(steering, throttle, max_conf)

        return Prediction(
            steering=steering,
            throttle=throttle,
            confidence=float(max_conf),
            agent_name="orchestrator",
        )

    def update_performance(
        self, agent_name: str, squared_error: float
    ) -> None:
        """Record a squared error for an agent (used during evaluation)."""
        history = self.performance_history[agent_name]
        history.append(squared_error)
        # Keep bounded
        if len(history) > self._history_window * 2:
            self.performance_history[agent_name] = history[
                -self._history_window :
            ]
