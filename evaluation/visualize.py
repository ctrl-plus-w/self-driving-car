import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_vs_ground_truth(
    true_steering: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str = "Steering Predictions vs Ground Truth",
    save_path: str | None = None,
    max_frames: int = 500,
) -> None:
    """Plot steering predictions from multiple agents against ground truth."""
    n = min(len(true_steering), max_frames)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, true_steering[:n], "k-", linewidth=2, alpha=0.7, label="Ground Truth")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, (name, preds) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        ax.plot(x, preds[:n], color=color, linewidth=1, alpha=0.7, label=name)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Steering Angle")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_metrics_comparison(
    results: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> None:
    """Bar chart comparing metrics across agents."""
    agents = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for ax, metric in zip(axes, metrics):
        values = [results[a][metric] for a in agents]
        bars = ax.bar(
            agents,
            values,
            color=colors[: len(agents)],
            alpha=0.8,
        )
        ax.set_title(metric)
        ax.set_ylabel("Value")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle("Agent Performance Comparison", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()
