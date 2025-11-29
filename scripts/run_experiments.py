#!/usr/bin/env python
"""Main experiment script for energy forecasting.

This script runs all experiments for the paper:
1. Baseline comparisons (ARIMA, Naive, MA, LSTM, CNN, Transformer)
2. Ablation studies (with/without cyclical encoding, with/without features)
3. Multi-dataset validation

Usage:
    python scripts/run_experiments.py --config configs/default.yaml
    python scripts/run_experiments.py --experiment baseline
    python scripts/run_experiments.py --experiment ablation
    python scripts/run_experiments.py --experiment all
"""

import argparse
import json
import logging
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import UCLDataLoader, ODREDataLoader, DataConfig
from src.features import TemporalFeatureEngine, CyclicalEncodingConfig
from src.models import (
    LSTMModel,
    CNNModel,
    TransformerModel,
    MetaClassifier,
    ARIMAModel,
    NaiveModel,
    MovingAverageModel,
    SeasonalNaiveModel,
)
from src.models.deep_models import DeepModelConfig
from src.models.ensemble import EnsembleConfig
from src.models.baselines import BaselineConfig
from src.training import Trainer, TrainerConfig
from src.evaluation import compute_all_metrics, MetricsResult, print_metrics_table, ResultsSaver
from src.visualization import create_paper_figures, plot_per_day_metrics
from src.features.temporal_encoding import compute_raw_vs_encoded_correlations

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_data_loaders(
    dataset: "TimeSeriesDataset",
    batch_size: int,
) -> DataLoader:
    """Create PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def run_baseline_experiments(
    train_data: np.ndarray,
    test_data: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, MetricsResult]:
    """Run baseline model experiments.

    Args:
        train_data: Training data array.
        test_data: Test data array.
        config: Configuration dictionary.

    Returns:
        Dictionary of model names to results.
    """
    logger.info("Running baseline experiments...")
    results = {}

    input_window = config["data"]["input_window"]
    output_horizon = config["data"]["output_horizon"]

    baseline_config = BaselineConfig(
        input_window=input_window,
        output_horizon=output_horizon,
    )

    # Naive baseline
    logger.info("Evaluating Naive baseline...")
    naive = NaiveModel(config=baseline_config)
    naive.fit(train_data)
    preds, actuals, _ = naive.evaluate(test_data)
    results["Naive"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["Naive"], "Naive")

    # Moving Average baseline
    logger.info("Evaluating Moving Average baseline...")
    ma = MovingAverageModel(window_size=7, config=baseline_config)
    ma.fit(train_data)
    preds, actuals, _ = ma.evaluate(test_data)
    results["MA(7)"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["MA(7)"], "Moving Average")

    # Seasonal Naive
    logger.info("Evaluating Seasonal Naive baseline...")
    snaive = SeasonalNaiveModel(seasonal_period=7, config=baseline_config)
    snaive.fit(train_data)
    preds, actuals, _ = snaive.evaluate(test_data)
    results["SNaive"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["SNaive"], "Seasonal Naive")

    # ARIMA
    logger.info("Evaluating ARIMA baseline...")
    try:
        arima = ARIMAModel(order=(5, 1, 0), config=baseline_config)
        arima.fit(train_data)
        preds, actuals, _ = arima.evaluate(test_data)
        results["ARIMA"] = compute_all_metrics(actuals, preds)
        print_metrics_table(results["ARIMA"], "ARIMA")
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}")

    return results


def run_deep_learning_experiments(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    use_features: bool = True,
    n_temporal_features: int = 0,
) -> Dict[str, MetricsResult]:
    """Run deep learning model experiments.

    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        config: Configuration dictionary.
        use_features: Whether models use temporal features.
        n_temporal_features: Number of temporal features.

    Returns:
        Dictionary of model names to results.
    """
    logger.info("Running deep learning experiments...")
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Common config
    model_config = DeepModelConfig(
        input_window=config["data"]["input_window"],
        output_horizon=config["data"]["output_horizon"],
        hidden_size=config["models"]["lstm"]["hidden_size"],
        num_layers=config["models"]["lstm"]["num_layers"],
        dropout=config["models"]["lstm"]["dropout"],
        n_temporal_features=n_temporal_features if use_features else 0,
    )

    trainer_config = TrainerConfig(
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
    )

    # LSTM
    logger.info("Training LSTM model...")
    lstm = LSTMModel(model_config).to(device)
    trainer = Trainer(lstm, trainer_config)
    trainer.fit(train_loader, val_loader)
    preds, actuals = trainer.predict(test_loader)
    results["LSTM"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["LSTM"], "LSTM")

    # CNN
    logger.info("Training CNN model...")
    cnn = CNNModel(model_config).to(device)
    trainer = Trainer(cnn, trainer_config)
    trainer.fit(train_loader, val_loader)
    preds, actuals = trainer.predict(test_loader)
    results["CNN"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["CNN"], "CNN")

    # Transformer
    logger.info("Training Transformer model...")
    transformer = TransformerModel(model_config).to(device)
    trainer = Trainer(transformer, trainer_config)
    trainer.fit(train_loader, val_loader)
    preds, actuals = trainer.predict(test_loader)
    results["Transformer"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["Transformer"], "Transformer")

    # Meta-Classifier (Ensemble)
    logger.info("Training Meta-Classifier ensemble...")
    ensemble_config = EnsembleConfig(
        input_window=config["data"]["input_window"],
        output_horizon=config["data"]["output_horizon"],
        lstm_hidden=config["models"]["lstm"]["hidden_size"],
        lstm_layers=config["models"]["lstm"]["num_layers"],
        n_temporal_features=n_temporal_features if use_features else 0,
    )
    mc = MetaClassifier(ensemble_config).to(device)
    trainer = Trainer(mc, trainer_config)
    trainer.fit(train_loader, val_loader)
    preds, actuals = trainer.predict(test_loader)
    results["MC"] = compute_all_metrics(actuals, preds)
    print_metrics_table(results["MC"], "Meta-Classifier")

    return results


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_loader_no_feat: DataLoader,
    val_loader_no_feat: DataLoader,
    test_loader_no_feat: DataLoader,
    config: Dict[str, Any],
    n_temporal_features: int,
) -> Dict[str, MetricsResult]:
    """Run ablation studies.

    Ablations:
    1. MC with cyclical encoding vs without
    2. MC with calendar features vs without
    3. LSTM only vs CNN only vs MC

    Args:
        Various data loaders and configurations.

    Returns:
        Dictionary of ablation results.
    """
    logger.info("Running ablation studies...")
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_config = TrainerConfig(
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
    )

    # Full model (with features)
    logger.info("Ablation: Full model with cyclical encoding...")
    ensemble_config = EnsembleConfig(
        input_window=config["data"]["input_window"],
        output_horizon=config["data"]["output_horizon"],
        lstm_hidden=config["models"]["lstm"]["hidden_size"],
        lstm_layers=config["models"]["lstm"]["num_layers"],
        n_temporal_features=n_temporal_features,
    )
    mc_full = MetaClassifier(ensemble_config).to(device)
    trainer = Trainer(mc_full, trainer_config)
    trainer.fit(train_loader, val_loader)
    preds, actuals = trainer.predict(test_loader)
    results["MC + Cyclical Enc."] = compute_all_metrics(actuals, preds)

    # Without features
    logger.info("Ablation: Model without temporal features...")
    ensemble_config_no_feat = EnsembleConfig(
        input_window=config["data"]["input_window"],
        output_horizon=config["data"]["output_horizon"],
        lstm_hidden=config["models"]["lstm"]["hidden_size"],
        lstm_layers=config["models"]["lstm"]["num_layers"],
        n_temporal_features=0,
    )
    mc_no_feat = MetaClassifier(ensemble_config_no_feat).to(device)
    trainer = Trainer(mc_no_feat, trainer_config)
    trainer.fit(train_loader_no_feat, val_loader_no_feat)
    preds, actuals = trainer.predict(test_loader_no_feat)
    results["MC - No Features"] = compute_all_metrics(actuals, preds)

    return results


def main():
    """Main entry point for experiments."""
    parser = argparse.ArgumentParser(description="Run energy forecasting experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["baseline", "deep", "ablation", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ucl", "odre", "both"],
        default="ucl",
        help="Which dataset to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    # Data configuration
    data_config = DataConfig(
        input_window=config["data"]["input_window"],
        output_horizon=config["data"]["output_horizon"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        normalize=config["data"]["normalize"],
    )

    # Load dataset
    if args.dataset in ["ucl", "both"]:
        logger.info("Loading UCL dataset...")
        ucl_loader = UCLDataLoader(config=data_config)
        ucl_loader.load("input/UCL_dataset/LD2011_2014.txt")

        # Extract temporal features
        feature_engine = TemporalFeatureEngine(
            CyclicalEncodingConfig(
                features=config["features"]["cyclical_features"],
                include_raw=config["features"]["include_raw"],
            )
        )
        features = feature_engine.fit_transform(
            ucl_loader.data.index,
            ucl_loader.data["consumption"],
        )
        features_array = features.values

        # Create datasets with features
        train_ds, val_ds, test_ds = ucl_loader.create_datasets(features_array)
        n_features = features_array.shape[1]

        # Create datasets without features (for ablation)
        train_ds_no_feat, val_ds_no_feat, test_ds_no_feat = ucl_loader.create_datasets()

        # Create data loaders
        batch_size = config["training"]["batch_size"]
        train_loader = create_data_loaders(train_ds, batch_size)
        val_loader = create_data_loaders(val_ds, batch_size)
        test_loader = create_data_loaders(test_ds, batch_size)

        train_loader_no_feat = create_data_loaders(train_ds_no_feat, batch_size)
        val_loader_no_feat = create_data_loaders(val_ds_no_feat, batch_size)
        test_loader_no_feat = create_data_loaders(test_ds_no_feat, batch_size)

        # Get raw data for baselines
        consumption = ucl_loader.get_consumption_series()
        if data_config.normalize:
            consumption = ucl_loader.normalize(consumption)

        n = len(consumption)
        train_end = int(n * data_config.train_ratio)
        val_end = int(n * (data_config.train_ratio + data_config.val_ratio))
        train_data = consumption[:train_end]
        test_data = consumption[val_end - data_config.input_window:]

        all_results = {}

        # Run experiments
        if args.experiment in ["baseline", "all"]:
            baseline_results = run_baseline_experiments(train_data, test_data, config)
            all_results.update(baseline_results)

        if args.experiment in ["deep", "all"]:
            deep_results = run_deep_learning_experiments(
                train_loader,
                val_loader,
                test_loader,
                config,
                use_features=True,
                n_temporal_features=n_features,
            )
            all_results.update(deep_results)

        if args.experiment in ["ablation", "all"]:
            ablation_results = run_ablation_study(
                train_loader,
                val_loader,
                test_loader,
                train_loader_no_feat,
                val_loader_no_feat,
                test_loader_no_feat,
                config,
                n_features,
            )
            all_results.update(ablation_results)

        # Save results using ResultsSaver
        logger.info("Saving results...")
        saver = ResultsSaver(str(output_dir))
        
        # Save all metrics in different formats
        saver.save_per_day_metrics(all_results)
        saver.save_overall_metrics(all_results)
        saver.save_wide_format_metrics(all_results)
        
        # Save feature correlations
        try:
            raw_vs_encoded = compute_raw_vs_encoded_correlations(
                ucl_loader.data.index,
                ucl_loader.data["consumption"],
            )
            saver.save_raw_vs_encoded_correlations(raw_vs_encoded)
            
            # Also save the feature engine correlations
            if feature_engine.feature_correlations is not None:
                saver.save_feature_correlations(feature_engine.feature_correlations)
        except Exception as e:
            logger.warning(f"Could not save correlations: {e}")
        
        # Save experiment config
        saver.save_experiment_config(config)
        
        # Also save the old format for backwards compatibility
        results_df = pd.DataFrame([
            {
                "Model": name,
                "MAE": r.mae,
                "RMSE": r.rmse,
                "MAPE": r.mape,
                **{f"MAE_Day{i+1}": r.per_day_mae[i] for i in range(7)},
                **{f"RMSE_Day{i+1}": r.per_day_rmse[i] for i in range(7)},
            }
            for name, r in all_results.items()
        ])
        results_df.to_csv(output_dir / "results.csv", index=False)

        # Create figures
        if all_results:
            create_paper_figures(all_results, str(output_dir / "figures"))

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        print(results_df.to_string(index=False))
        
        # Print saved files summary
        logger.info("\nSaved CSV files:")
        logger.info(f"  - {output_dir}/metrics/per_day_metrics.csv")
        logger.info(f"  - {output_dir}/metrics/overall_metrics.csv")
        logger.info(f"  - {output_dir}/metrics/metrics_wide.csv")
        logger.info(f"  - {output_dir}/correlations/feature_correlations.csv")
        logger.info(f"  - {output_dir}/correlations/raw_vs_encoded_correlations.csv")
        logger.info(f"  - {output_dir}/experiment_config.json")

        logger.info(f"\nAll results saved to {output_dir}")

    logger.info("Experiments completed!")


if __name__ == "__main__":
    main()

