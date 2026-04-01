"""Agent 3 – Classical CV feature extraction + XGBoost gradient boosting.

Uses hand-crafted lane and colour features (no deep learning) fed into
two separate XGBRegressor models: one for steering, one for throttle.
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from agents.base_agent import BaseAgent, Prediction
from config.settings import CAMERA_STEERING_OFFSET, TRAIN_SPLIT, VAL_SPLIT
from models.feature_extractor import extract_all_features


class ClassicalAgent(BaseAgent):
    """XGBoost-based driving agent using classical CV features."""

    name: str = "classical_xgboost"

    def __init__(self) -> None:
        self.steering_model: XGBRegressor | None = None
        self.throttle_model: XGBRegressor | None = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        speed: float,
        history: list[np.ndarray] | None = None,
    ) -> Prediction:
        """Extract features from *image* and run through the two XGBoost models."""
        if self.steering_model is None or self.throttle_model is None:
            raise RuntimeError(
                "Models not loaded. Call load() or train() before predict()."
            )

        features, lane_confidence = extract_all_features(image, speed)
        X = features.reshape(1, -1)

        steering = float(self.steering_model.predict(X)[0])
        throttle = float(self.throttle_model.predict(X)[0])

        # Clamp to valid ranges
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return Prediction(
            steering=steering,
            throttle=throttle,
            confidence=lane_confidence,
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data_path: str, **kwargs) -> dict:
        """Train steering and throttle XGBRegressors from a driving-log CSV.

        Parameters
        ----------
        data_path : str
            Path to the CSV driving log.
        **kwargs
            image_root : str – directory containing images (defaults to
                               parent of *data_path*).

        Returns
        -------
        dict with training / validation MSE for both targets.
        """
        image_root = Path(kwargs.get("image_root", Path(data_path).parent))

        # ---- Load CSV ------------------------------------------------
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip()

        # ---- Build feature matrix & targets --------------------------
        features_list: list[np.ndarray] = []
        steering_list: list[float] = []
        throttle_list: list[float] = []

        for _, row in df.iterrows():
            center_path = str(row["centercam"]).strip()
            full_path = image_root / Path(center_path).name
            if not full_path.exists():
                full_path = Path(center_path)

            img = cv2.imread(str(full_path))
            if img is None:
                continue

            steering = float(row["steering_angle"])
            throttle = float(row["throttle"])
            speed = float(row["speed"]) if "speed" in df.columns else 0.0

            feat, _ = extract_all_features(img, speed)
            features_list.append(feat)
            steering_list.append(steering)
            throttle_list.append(throttle)

        X = np.stack(features_list)
        y_steer = np.array(steering_list)
        y_throttle = np.array(throttle_list)

        # ---- Train / validation split --------------------------------
        test_size = 1.0 - TRAIN_SPLIT
        X_train, X_val, ys_train, ys_val, yt_train, yt_val = train_test_split(
            X, y_steer, y_throttle, test_size=test_size, random_state=42
        )

        # ---- Fit XGBoost models --------------------------------------
        xgb_params = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )

        self.steering_model = XGBRegressor(**xgb_params)
        self.steering_model.fit(
            X_train,
            ys_train,
            eval_set=[(X_val, ys_val)],
            verbose=False,
        )

        self.throttle_model = XGBRegressor(**xgb_params)
        self.throttle_model.fit(
            X_train,
            yt_train,
            eval_set=[(X_val, yt_val)],
            verbose=False,
        )

        # ---- Compute validation metrics ------------------------------
        steer_mse = float(np.mean((self.steering_model.predict(X_val) - ys_val) ** 2))
        throttle_mse = float(
            np.mean((self.throttle_model.predict(X_val) - yt_val) ** 2)
        )

        metrics = {
            "val_steering_mse": steer_mse,
            "val_throttle_mse": throttle_mse,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, checkpoint_path: str) -> None:
        """Save both XGBoost models to *checkpoint_path* using joblib.

        Creates two files:
            <checkpoint_path>_steering.joblib
            <checkpoint_path>_throttle.joblib
        """
        joblib.dump(self.steering_model, f"{checkpoint_path}_steering.joblib")
        joblib.dump(self.throttle_model, f"{checkpoint_path}_throttle.joblib")

    def load(self, checkpoint_path: str) -> None:
        """Load both XGBoost models from *checkpoint_path*."""
        self.steering_model = joblib.load(f"{checkpoint_path}_steering.joblib")
        self.throttle_model = joblib.load(f"{checkpoint_path}_throttle.joblib")
