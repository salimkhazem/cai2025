"""Evaluation metrics for time series forecasting.

This module provides standard metrics for evaluating forecasting models:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

Metrics are computed both overall and per-day for the forecast horizon.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MetricsResult:
    """Container for evaluation metrics.

    Attributes:
        mae: Mean Absolute Error (overall).
        rmse: Root Mean Squared Error (overall).
        mape: Mean Absolute Percentage Error (overall).
        per_day_mae: MAE for each forecast day.
        per_day_rmse: RMSE for each forecast day.
        per_day_mape: MAPE for each forecast day.
        n_samples: Number of samples evaluated.
    """

    mae: float
    rmse: float
    mape: float
    per_day_mae: List[float]
    per_day_rmse: List[float]
    per_day_mape: List[float]
    n_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary with all metrics.
        """
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "per_day_mae": self.per_day_mae,
            "per_day_rmse": self.per_day_rmse,
            "per_day_mape": self.per_day_mape,
            "n_samples": self.n_samples,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with per-day metrics.

        Returns:
            DataFrame with metrics for each forecast day.
        """
        n_days = len(self.per_day_mae)
        return pd.DataFrame({
            "Day": range(1, n_days + 1),
            "MAE": self.per_day_mae,
            "RMSE": self.per_day_rmse,
            "MAPE": self.per_day_mape,
        })


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    MAE = (1/n) * Σ|y_true - y_pred|

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    RMSE = sqrt((1/n) * Σ(y_true - y_pred)²)

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Root Mean Squared Error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute Mean Absolute Percentage Error.

    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|

    Args:
        y_true: True values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.

    Returns:
        Mean Absolute Percentage Error (in percentage).
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon))


def compute_per_day_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, List[float]]:
    """Compute metrics for each forecast day.

    Args:
        y_true: True values of shape (n_samples, output_horizon).
        y_pred: Predicted values of shape (n_samples, output_horizon).

    Returns:
        Dictionary with lists of per-day MAE, RMSE, MAPE.
    """
    n_days = y_true.shape[1]

    mae_list = []
    rmse_list = []
    mape_list = []

    for day in range(n_days):
        mae_list.append(compute_mae(y_true[:, day], y_pred[:, day]))
        rmse_list.append(compute_rmse(y_true[:, day], y_pred[:, day]))
        mape_list.append(compute_mape(y_true[:, day], y_pred[:, day]))

    return {
        "mae": mae_list,
        "rmse": rmse_list,
        "mape": mape_list,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> MetricsResult:
    """Compute all metrics (overall and per-day).

    Args:
        y_true: True values of shape (n_samples, output_horizon).
        y_pred: Predicted values of shape (n_samples, output_horizon).

    Returns:
        MetricsResult with all computed metrics.
    """
    # Overall metrics
    overall_mae = compute_mae(y_true, y_pred)
    overall_rmse = compute_rmse(y_true, y_pred)
    overall_mape = compute_mape(y_true, y_pred)

    # Per-day metrics
    per_day = compute_per_day_metrics(y_true, y_pred)

    return MetricsResult(
        mae=overall_mae,
        rmse=overall_rmse,
        mape=overall_mape,
        per_day_mae=per_day["mae"],
        per_day_rmse=per_day["rmse"],
        per_day_mape=per_day["mape"],
        n_samples=len(y_true),
    )


def create_results_table(
    results: Dict[str, MetricsResult],
    output_horizon: int = 7,
) -> pd.DataFrame:
    """Create a comparison table of multiple models.

    Args:
        results: Dictionary mapping model names to MetricsResult.
        output_horizon: Number of forecast days.

    Returns:
        DataFrame with model comparison.
    """
    rows = []

    for model_name, metrics in results.items():
        for day in range(output_horizon):
            rows.append({
                "Model": model_name,
                "Day": day + 1,
                "MAE": metrics.per_day_mae[day],
                "RMSE": metrics.per_day_rmse[day],
                "MAPE": metrics.per_day_mape[day],
            })

        # Add average row
        rows.append({
            "Model": model_name,
            "Day": "Avg.",
            "MAE": metrics.mae,
            "RMSE": metrics.rmse,
            "MAPE": metrics.mape,
        })

    return pd.DataFrame(rows)


def print_metrics_table(results: MetricsResult, model_name: str = "Model") -> None:
    """Print formatted metrics table.

    Args:
        results: MetricsResult to display.
        model_name: Name of the model.
    """
    print(f"\n{model_name} - Forecasting Results")
    print("=" * 50)
    print(f"{'Day':<8} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    print("-" * 50)

    for day in range(len(results.per_day_mae)):
        print(
            f"{day + 1:<8} "
            f"{results.per_day_mae[day]:>10.4f} "
            f"{results.per_day_rmse[day]:>10.4f} "
            f"{results.per_day_mape[day]:>10.2f}%"
        )

    print("-" * 50)
    print(
        f"{'Avg.':<8} "
        f"{results.mae:>10.4f} "
        f"{results.rmse:>10.4f} "
        f"{results.mape:>10.2f}%"
    )
    print("=" * 50)


def compute_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Compute confidence intervals for metrics using bootstrap.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        confidence: Confidence level (e.g., 0.95 for 95%).
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Dictionary with metric confidence intervals.
    """
    n_samples = len(y_true)
    alpha = (1 - confidence) / 2

    mae_samples = []
    rmse_samples = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]

        mae_samples.append(compute_mae(y_t, y_p))
        rmse_samples.append(compute_rmse(y_t, y_p))

    return {
        "mae": {
            "mean": np.mean(mae_samples),
            "lower": np.percentile(mae_samples, alpha * 100),
            "upper": np.percentile(mae_samples, (1 - alpha) * 100),
        },
        "rmse": {
            "mean": np.mean(rmse_samples),
            "lower": np.percentile(rmse_samples, alpha * 100),
            "upper": np.percentile(rmse_samples, (1 - alpha) * 100),
        },
    }

