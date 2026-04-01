import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from agents.base_agent import BaseAgent, Prediction
from config.settings import TRAIN_SPLIT, VAL_SPLIT
from evaluation.metrics import compute_all_metrics
from orchestrator.orchestrator import Orchestrator


def evaluate_agent(
    agent: BaseAgent,
    csv_path: str | Path,
    image_root: str | Path,
    split: str = "test",
) -> dict[str, float]:
    """Evaluate a single agent on the given data split. Returns metrics dict."""
    image_root = Path(image_root)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    if split == "train":
        df = df.iloc[:train_end]
    elif split == "val":
        df = df.iloc[train_end:val_end]
    elif split == "test":
        df = df.iloc[val_end:]

    df = df.reset_index(drop=True)

    pred_steerings = []
    pred_throttles = []
    true_steerings = []
    true_throttles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {agent.name}"):
        img_path = image_root / Path(str(row["centercam"]).strip()).name
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        speed = float(row["speed"])
        prediction = agent.predict(image, speed)

        pred_steerings.append(prediction.steering)
        pred_throttles.append(prediction.throttle)
        true_steerings.append(float(row["steering_angle"]))
        true_throttles.append(float(row["throttle"]))

    return compute_all_metrics(
        np.array(pred_steerings),
        np.array(pred_throttles),
        np.array(true_steerings),
        np.array(true_throttles),
    )


def evaluate_orchestrator(
    orchestrator: Orchestrator,
    csv_path: str | Path,
    image_root: str | Path,
    split: str = "test",
) -> dict[str, dict[str, float]]:
    """Evaluate all agents and the orchestrator. Returns metrics per agent + orchestrator."""
    results = {}

    # Evaluate each agent individually
    for agent in orchestrator.agents:
        results[agent.name] = evaluate_agent(agent, csv_path, image_root, split)

    # Evaluate orchestrator ensemble
    image_root = Path(image_root)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    if split == "test":
        df = df.iloc[val_end:]
    df = df.reset_index(drop=True)

    pred_steerings = []
    pred_throttles = []
    true_steerings = []
    true_throttles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval Orchestrator"):
        img_path = image_root / Path(str(row["centercam"]).strip()).name
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        speed = float(row["speed"])
        true_steer = float(row["steering_angle"])
        true_throt = float(row["throttle"])

        prediction = orchestrator.predict(image, speed)

        # Update per-agent performance history for adaptive dampening
        for agent_pred in orchestrator._last_predictions:
            steering_se = (agent_pred.steering - true_steer) ** 2
            throttle_se = (agent_pred.throttle - true_throt) ** 2
            orchestrator.update_performance(
                agent_pred.agent_name, (steering_se + throttle_se) / 2
            )

        pred_steerings.append(prediction.steering)
        pred_throttles.append(prediction.throttle)
        true_steerings.append(true_steer)
        true_throttles.append(true_throt)

    results["orchestrator"] = compute_all_metrics(
        np.array(pred_steerings),
        np.array(pred_throttles),
        np.array(true_steerings),
        np.array(true_throttles),
    )

    return results
