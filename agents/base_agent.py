from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    steering: float       # [-1, 1]
    throttle: float       # [0, 1]
    confidence: float     # [0, 1]
    agent_name: str


class BaseAgent(ABC):
    """Abstract base class that every prediction agent must implement."""

    name: str = "base"

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        speed: float,
        history: list[np.ndarray] | None = None,
    ) -> Prediction:
        """Given current image, speed, and optional frame history, return prediction."""
        ...

    @abstractmethod
    def train(self, data_path: str, **kwargs) -> dict:
        """Train the agent's model. Returns training metrics dict."""
        ...

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """Load a trained model from disk."""
        ...

    @abstractmethod
    def save(self, checkpoint_path: str) -> None:
        """Save the trained model to disk."""
        ...
