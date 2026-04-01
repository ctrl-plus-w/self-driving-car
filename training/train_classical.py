"""Training script for Agent 3 (Classical CV + XGBoost).

Usage
-----
    python -m training.train_classical --csv data/raw/driving_log.csv \
                                       --image-root data/raw/IMG
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.classical_agent import ClassicalAgent
from config.settings import CHECKPOINTS_DIR, TRAIN_SPLIT
from models.feature_extractor import extract_all_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost steering/throttle models on classical CV features."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the driving-log CSV file.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Directory containing the images. Defaults to the parent dir of --csv.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Base path for saving model checkpoints (without extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    image_root = Path(args.image_root) if args.image_root else csv_path.parent
    checkpoint = args.checkpoint or str(CHECKPOINTS_DIR / "classical_xgboost")

    # ------------------------------------------------------------------
    # 1. Load CSV and extract features
    # ------------------------------------------------------------------
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    features_list: list[np.ndarray] = []
    steering_list: list[float] = []
    throttle_list: list[float] = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        center_path = str(row["centercam"]).strip()
        full_path = image_root / Path(center_path).name
        if not full_path.exists():
            full_path = Path(center_path)

        img = cv2.imread(str(full_path))
        if img is None:
            skipped += 1
            continue

        speed = float(row["speed"]) if "speed" in df.columns else 0.0
        feat, _ = extract_all_features(img, speed)

        features_list.append(feat)
        steering_list.append(float(row["steering_angle"]))
        throttle_list.append(float(row["throttle"]))

    print(
        f"Extracted {len(features_list)} samples "
        f"({skipped} images skipped)."
    )

    if len(features_list) == 0:
        print("ERROR: No valid samples found. Check image paths.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Train the agent
    # ------------------------------------------------------------------
    agent = ClassicalAgent()
    print("Training XGBoost models ...")
    metrics = agent.train(str(csv_path), image_root=str(image_root))

    # ------------------------------------------------------------------
    # 3. Report validation metrics
    # ------------------------------------------------------------------
    print("\n--- Validation Results ---")
    print(f"  Steering MSE : {metrics['val_steering_mse']:.6f}")
    print(f"  Throttle MSE : {metrics['val_throttle_mse']:.6f}")
    print(f"  Train samples: {metrics['train_samples']}")
    print(f"  Val samples  : {metrics['val_samples']}")

    # ------------------------------------------------------------------
    # 4. Save models
    # ------------------------------------------------------------------
    agent.save(checkpoint)
    print(f"\nModels saved to {checkpoint}_steering.joblib / {checkpoint}_throttle.joblib")


if __name__ == "__main__":
    main()
