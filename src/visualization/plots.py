"""Visualization functions for paper figures and analysis.

This module provides publication-quality plotting functions for:
- Metric comparisons across models
- Per-day forecasting performance
- Feature correlation analysis
- Training curves
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..evaluation.metrics import MetricsResult

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot predictions against actual values.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        dates: Optional datetime index for x-axis.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    x = dates if dates is not None else range(len(y_true))

    ax.plot(x, y_true, label="Actual", color="blue", linewidth=1.5)
    ax.plot(x, y_pred, label="Predicted", color="red", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_per_day_metrics(
    results: Dict[str, MetricsResult],
    metric: str = "mae",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-day metrics comparison across models.

    This creates figures similar to Figure 5 in the paper.

    Args:
        results: Dictionary mapping model names to MetricsResult.
        metric: Metric to plot ('mae', 'rmse', 'mape').
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_key = f"per_day_{metric}"
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for i, (model_name, metrics) in enumerate(results.items()):
        values = getattr(metrics, metric_key)
        days = range(1, len(values) + 1)
        ax.plot(
            days,
            values,
            marker="o",
            label=model_name,
            color=colors[i],
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Forecast Day")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} per Forecast Day")
    ax.set_xticks(range(1, 8))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_metrics_comparison(
    results: Dict[str, MetricsResult],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create bar chart comparing overall metrics across models.

    Args:
        results: Dictionary mapping model names to MetricsResult.
        metrics: List of metrics to include.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    metrics = metrics or ["mae", "rmse"]
    models = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, metrics):
        values = [getattr(results[m], metric) for m in models]
        bars = ax.bar(models, values, color=colors)

        ax.set_ylabel(metric.upper())
        ax.set_title(f"Overall {metric.upper()}")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_correlation_heatmap(
    correlations: pd.DataFrame,
    title: str = "Feature Correlations with Target",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot correlation heatmap for features.

    Args:
        correlations: DataFrame with feature correlations.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a matrix for heatmap
    if "correlation" in correlations.columns:
        data = correlations.set_index("feature")["correlation"].to_frame().T
    else:
        data = correlations

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_ablation_comparison(
    baseline_results: MetricsResult,
    ablation_results: Dict[str, MetricsResult],
    metric: str = "mae",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ablation study comparison.

    Args:
        baseline_results: Full model results.
        ablation_results: Results for ablated variants.
        metric: Metric to compare.
        save_path: Optional path to save.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Combine baseline with ablations
    all_results = {"Full Model": baseline_results, **ablation_results}

    models = list(all_results.keys())
    values = [getattr(r, metric) for r in all_results.values()]

    colors = ["green"] + ["red"] * len(ablation_results)
    bars = ax.bar(models, values, color=colors, edgecolor="black")

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study - {metric.upper()} Comparison")
    ax.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="Full Model"),
        Patch(facecolor="red", label="Ablated"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"Saved figure to {save_path}")

    return fig


def create_paper_figures(
    results: Dict[str, MetricsResult],
    output_dir: str = "figures",
) -> None:
    """Create all figures for the paper.

    Args:
        results: Dictionary mapping model names to results.
        output_dir: Directory to save figures.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Figure 5a: MAE per day
    plot_per_day_metrics(
        results,
        metric="mae",
        title="(a) MAE per Day per Model",
        save_path=str(output_path / "fig5a_mae_per_day.pdf"),
    )

    # Figure 5b: RMSE per day
    plot_per_day_metrics(
        results,
        metric="rmse",
        title="(b) RMSE per Day per Model",
        save_path=str(output_path / "fig5b_rmse_per_day.pdf"),
    )

    # Overall comparison
    plot_metrics_comparison(
        results,
        metrics=["mae", "rmse"],
        save_path=str(output_path / "overall_comparison.pdf"),
    )

    logger.info(f"Created all paper figures in {output_dir}")


def create_results_latex_table(
    results: Dict[str, MetricsResult],
    output_path: Optional[str] = None,
) -> str:
    """Create LaTeX table for paper.

    Args:
        results: Dictionary mapping model names to results.
        output_path: Optional path to save the table.

    Returns:
        LaTeX table string.
    """
    # Build the table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{7-Day Forecasting Results: MAE and RMSE for Each Model}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{c" + "cc" * len(results) + "}")
    lines.append(r"\toprule")

    # Header
    header = "Day"
    for model in results:
        header += f" & \\multicolumn{{2}}{{c}}{{{model}}}"
    header += r" \\"
    lines.append(header)

    # Sub-header
    subheader = ""
    for _ in results:
        subheader += " & MAE & RMSE"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    # Data rows
    n_days = len(list(results.values())[0].per_day_mae)
    for day in range(n_days):
        row = str(day + 1)
        for metrics in results.values():
            row += f" & {metrics.per_day_mae[day]:.3f} & {metrics.per_day_rmse[day]:.3f}"
        row += r" \\"
        lines.append(row)

    # Average row
    lines.append(r"\midrule")
    avg_row = "Avg."
    for metrics in results.values():
        avg_row += f" & {metrics.mae:.3f} & {metrics.rmse:.3f}"
    avg_row += r" \\"
    lines.append(avg_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Saved LaTeX table to {output_path}")

    return table

