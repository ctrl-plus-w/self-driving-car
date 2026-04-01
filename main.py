"""Multi-Agent Self-Driving Car — Steering & Throttle Prediction.

Entry point for training, evaluating, and running the orchestrator.
"""

import argparse
import json

from agents.pilotnet_agent import PilotNetAgent
from agents.resnet_agent import ResNetAgent
from agents.classical_agent import ClassicalAgent
from agents.temporal_agent import TemporalAgent
from orchestrator.orchestrator import Orchestrator
from evaluation.evaluate import evaluate_agent, evaluate_orchestrator
from evaluation.visualize import plot_metrics_comparison


def build_agents() -> list:
    """Instantiate all 4 agents."""
    return [
        PilotNetAgent(),
        ResNetAgent(),
        ClassicalAgent(),
        TemporalAgent(),
    ]


def cmd_train(args: argparse.Namespace) -> None:
    """Train a specific agent or all agents."""
    agents = build_agents()
    agent_map = {a.name: a for a in agents}

    targets = [args.agent] if args.agent != "all" else list(agent_map.keys())

    for name in targets:
        if name not in agent_map:
            print(f"Unknown agent: {name}. Available: {list(agent_map.keys())}")
            continue
        agent = agent_map[name]
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        metrics = agent.train(
            data_path=args.data,
            csv_path=args.csv,
            image_root=args.images,
        )
        print(f"Training complete. Metrics: {metrics}")
        agent.save(f"checkpoints/{name}.pt")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate agents and the orchestrator on test data."""
    agents = build_agents()

    # Load trained checkpoints
    for agent in agents:
        checkpoint = f"checkpoints/{agent.name}.pt"
        try:
            agent.load(checkpoint)
            print(f"Loaded {agent.name} from {checkpoint}")
        except FileNotFoundError:
            print(f"Warning: No checkpoint for {agent.name}, using untrained model")

    orchestrator = Orchestrator(agents)

    print("\nEvaluating on test set...")
    results = evaluate_orchestrator(
        orchestrator,
        csv_path=args.csv,
        image_root=args.images,
        split="test",
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

    if args.plot:
        plot_metrics_comparison(results, save_path="evaluation_results.png")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Run the orchestrator on a single image."""
    import cv2

    agents = build_agents()
    for agent in agents:
        try:
            agent.load(f"checkpoints/{agent.name}.pt")
        except FileNotFoundError:
            print(f"Warning: No checkpoint for {agent.name}")

    orchestrator = Orchestrator(agents)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    result = orchestrator.predict(image, speed=args.speed)
    print(f"Steering: {result.steering:.4f}")
    print(f"Throttle: {result.throttle:.4f}")
    print(f"Confidence: {result.confidence:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Self-Driving Car Prediction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = subparsers.add_parser("train", help="Train agents")
    train_parser.add_argument("--agent", default="all", help="Agent name or 'all'")
    train_parser.add_argument("--csv", required=True, help="Path to driving CSV")
    train_parser.add_argument("--images", required=True, help="Path to image directory")
    train_parser.add_argument("--data", default=".", help="Base data path")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agents")
    eval_parser.add_argument("--csv", required=True, help="Path to driving CSV")
    eval_parser.add_argument("--images", required=True, help="Path to image directory")
    eval_parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    eval_parser.add_argument("--output", help="Save results JSON to file")

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Run prediction on image")
    pred_parser.add_argument("--image", required=True, help="Path to input image")
    pred_parser.add_argument("--speed", type=float, default=15.0, help="Current speed")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
