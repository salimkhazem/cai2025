"""Utilities for saving experiment results to CSV files.

This module provides functions to save experimental results in
formats suitable for post-processing, plotting, and analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .metrics import MetricsResult

logger = logging.getLogger(__name__)


class ResultsSaver:
    """Saves experiment results to various CSV formats.

    This class provides methods to save:
    - Per-day metrics for all models
    - Overall metrics summary
    - Predictions vs actuals
    - Feature correlations
    - Training history

    Example:
        >>> saver = ResultsSaver("results")
        >>> saver.save_all_metrics(results_dict)
        >>> saver.save_predictions(predictions, actuals, model_name)
    """

    def __init__(self, output_dir: str = "results") -> None:
        """Initialize the results saver.

        Args:
            output_dir: Directory to save results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "correlations").mkdir(exist_ok=True)
        (self.output_dir / "history").mkdir(exist_ok=True)

    def save_per_day_metrics(
        self,
        results: Dict[str, MetricsResult],
        filename: str = "per_day_metrics.csv",
    ) -> str:
        """Save per-day metrics for all models to CSV.

        Creates a long-format DataFrame suitable for plotting with
        seaborn/matplotlib.

        Args:
            results: Dictionary mapping model names to MetricsResult.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        rows = []
        for model_name, metrics in results.items():
            n_days = len(metrics.per_day_mae)
            for day in range(n_days):
                rows.append({
                    "Model": model_name,
                    "Day": day + 1,
                    "MAE": metrics.per_day_mae[day],
                    "RMSE": metrics.per_day_rmse[day],
                    "MAPE": metrics.per_day_mape[day],
                })

        df = pd.DataFrame(rows)
        filepath = self.output_dir / "metrics" / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved per-day metrics to {filepath}")
        return str(filepath)

    def save_overall_metrics(
        self,
        results: Dict[str, MetricsResult],
        filename: str = "overall_metrics.csv",
    ) -> str:
        """Save overall metrics summary for all models.

        Args:
            results: Dictionary mapping model names to MetricsResult.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        rows = []
        for model_name, metrics in results.items():
            rows.append({
                "Model": model_name,
                "MAE": metrics.mae,
                "RMSE": metrics.rmse,
                "MAPE": metrics.mape,
                "N_Samples": metrics.n_samples,
            })

        df = pd.DataFrame(rows)
        filepath = self.output_dir / "metrics" / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved overall metrics to {filepath}")
        return str(filepath)

    def save_wide_format_metrics(
        self,
        results: Dict[str, MetricsResult],
        filename: str = "metrics_wide.csv",
    ) -> str:
        """Save metrics in wide format (one row per model).

        Each row contains all metrics for one model, with columns
        for each day's MAE, RMSE, etc.

        Args:
            results: Dictionary mapping model names to MetricsResult.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        rows = []
        for model_name, metrics in results.items():
            row = {
                "Model": model_name,
                "MAE_Avg": metrics.mae,
                "RMSE_Avg": metrics.rmse,
                "MAPE_Avg": metrics.mape,
            }
            # Add per-day metrics
            for i, (mae, rmse, mape) in enumerate(zip(
                metrics.per_day_mae,
                metrics.per_day_rmse,
                metrics.per_day_mape
            )):
                row[f"MAE_Day{i+1}"] = mae
                row[f"RMSE_Day{i+1}"] = rmse
                row[f"MAPE_Day{i+1}"] = mape
            rows.append(row)

        df = pd.DataFrame(rows)
        filepath = self.output_dir / "metrics" / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved wide format metrics to {filepath}")
        return str(filepath)

    def save_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        model_name: str,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> str:
        """Save predictions and actuals for a model.

        Args:
            predictions: Predicted values (n_samples, horizon).
            actuals: Actual values (n_samples, horizon).
            model_name: Name of the model.
            dates: Optional datetime index for samples.

        Returns:
            Path to saved file.
        """
        n_samples, horizon = predictions.shape
        
        # Create DataFrame
        data = {"Sample": range(n_samples)}
        if dates is not None:
            data["Date"] = dates[:n_samples]
        
        for day in range(horizon):
            data[f"Pred_Day{day+1}"] = predictions[:, day]
            data[f"Actual_Day{day+1}"] = actuals[:, day]
            data[f"Error_Day{day+1}"] = predictions[:, day] - actuals[:, day]
        
        df = pd.DataFrame(data)
        
        # Safe filename
        safe_name = model_name.replace(" ", "_").replace("+", "plus").replace("-", "minus")
        filepath = self.output_dir / "predictions" / f"{safe_name}_predictions.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved predictions for {model_name} to {filepath}")
        return str(filepath)

    def save_feature_correlations(
        self,
        correlations: pd.DataFrame,
        filename: str = "feature_correlations.csv",
    ) -> str:
        """Save feature correlation analysis.

        Args:
            correlations: DataFrame with feature correlations.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        filepath = self.output_dir / "correlations" / filename
        correlations.to_csv(filepath, index=False)
        logger.info(f"Saved feature correlations to {filepath}")
        return str(filepath)

    def save_raw_vs_encoded_correlations(
        self,
        comparison: pd.DataFrame,
        filename: str = "raw_vs_encoded_correlations.csv",
    ) -> str:
        """Save comparison of raw vs encoded feature correlations.

        Args:
            comparison: DataFrame comparing raw and encoded correlations.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        filepath = self.output_dir / "correlations" / filename
        comparison.to_csv(filepath, index=False)
        logger.info(f"Saved raw vs encoded correlations to {filepath}")
        return str(filepath)

    def save_training_history(
        self,
        history: Dict[str, List[float]],
        model_name: str,
    ) -> str:
        """Save training history for a model.

        Args:
            history: Dictionary with train_loss, val_loss, etc.
            model_name: Name of the model.

        Returns:
            Path to saved file.
        """
        df = pd.DataFrame(history)
        df["Epoch"] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ["Epoch"] + [c for c in df.columns if c != "Epoch"]
        df = df[cols]
        
        safe_name = model_name.replace(" ", "_").replace("+", "plus").replace("-", "minus")
        filepath = self.output_dir / "history" / f"{safe_name}_history.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved training history for {model_name} to {filepath}")
        return str(filepath)

    def save_ablation_results(
        self,
        baseline_name: str,
        baseline_results: MetricsResult,
        ablation_results: Dict[str, MetricsResult],
        filename: str = "ablation_results.csv",
    ) -> str:
        """Save ablation study results.

        Args:
            baseline_name: Name of the baseline (full) model.
            baseline_results: Results for the baseline model.
            ablation_results: Dictionary of ablated model results.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        rows = []
        
        # Add baseline
        rows.append({
            "Configuration": baseline_name,
            "MAE": baseline_results.mae,
            "RMSE": baseline_results.rmse,
            "MAPE": baseline_results.mape,
            "Delta_MAE_Pct": 0.0,
            "Delta_RMSE_Pct": 0.0,
            "Is_Baseline": True,
        })
        
        # Add ablations
        for name, metrics in ablation_results.items():
            delta_mae = (metrics.mae - baseline_results.mae) / baseline_results.mae * 100
            delta_rmse = (metrics.rmse - baseline_results.rmse) / baseline_results.rmse * 100
            rows.append({
                "Configuration": name,
                "MAE": metrics.mae,
                "RMSE": metrics.rmse,
                "MAPE": metrics.mape,
                "Delta_MAE_Pct": delta_mae,
                "Delta_RMSE_Pct": delta_rmse,
                "Is_Baseline": False,
            })
        
        df = pd.DataFrame(rows)
        filepath = self.output_dir / "metrics" / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved ablation results to {filepath}")
        return str(filepath)

    def save_experiment_config(
        self,
        config: Dict[str, Any],
        filename: str = "experiment_config.json",
    ) -> str:
        """Save experiment configuration.

        Args:
            config: Configuration dictionary.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        # Add timestamp
        config_with_meta = {
            "timestamp": datetime.now().isoformat(),
            **config,
        }
        
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(config_with_meta, f, indent=2, default=str)
        logger.info(f"Saved experiment config to {filepath}")
        return str(filepath)

    def save_all(
        self,
        results: Dict[str, MetricsResult],
        predictions: Optional[Dict[str, tuple]] = None,
        correlations: Optional[pd.DataFrame] = None,
        histories: Optional[Dict[str, Dict]] = None,
        config: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """Save all results at once.

        Args:
            results: Model results dictionary.
            predictions: Optional dict of (predictions, actuals) per model.
            correlations: Optional feature correlations DataFrame.
            histories: Optional training histories per model.
            config: Optional experiment configuration.

        Returns:
            Dictionary of saved file paths.
        """
        saved_files = {}
        
        # Save metrics
        saved_files["per_day_metrics"] = self.save_per_day_metrics(results)
        saved_files["overall_metrics"] = self.save_overall_metrics(results)
        saved_files["wide_metrics"] = self.save_wide_format_metrics(results)
        
        # Save predictions
        if predictions:
            for model_name, (preds, acts) in predictions.items():
                key = f"predictions_{model_name}"
                saved_files[key] = self.save_predictions(preds, acts, model_name)
        
        # Save correlations
        if correlations is not None:
            saved_files["correlations"] = self.save_feature_correlations(correlations)
        
        # Save histories
        if histories:
            for model_name, history in histories.items():
                key = f"history_{model_name}"
                saved_files[key] = self.save_training_history(history, model_name)
        
        # Save config
        if config:
            saved_files["config"] = self.save_experiment_config(config)
        
        logger.info(f"Saved {len(saved_files)} result files")
        return saved_files

