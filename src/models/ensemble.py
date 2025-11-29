"""Ensemble forecasting model using Meta-Classifier architecture.

This module implements the Meta-Classifier (MC) ensemble approach that
combines predictions from multiple base models (LSTM, CNN) using a
set of MLP regressors - one for each forecast day.

The key insight is that different models may excel at different
forecast horizons, and the meta-learner learns to optimally weight
their contributions for each horizon.

Architecture:
    Base Models (LSTM, CNN) -> Intermediate Features -> 
    Meta-Classifiers (MLPs per day) -> Final Predictions
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from .deep_models import LSTMModel, CNNModel, DeepModelConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble model.

    Attributes:
        input_window: Number of time steps in input sequence.
        output_horizon: Number of days to forecast.
        lstm_hidden: Hidden size for LSTM model.
        lstm_layers: Number of LSTM layers.
        cnn_filters: List of filter sizes for CNN.
        n_temporal_features: Number of temporal features.
        dropout: Dropout probability.
        meta_hidden_layers: Hidden layer sizes for meta-MLP.
        meta_activation: Activation function for meta-MLP.
        meta_solver: Optimizer for meta-MLP ('adam', 'sgd').
        meta_learning_rate: Learning rate for meta-MLP.
    """

    input_window: int = 120
    output_horizon: int = 7
    lstm_hidden: int = 64
    lstm_layers: int = 2
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    n_temporal_features: int = 0
    dropout: float = 0.2
    meta_hidden_layers: Tuple[int, ...] = (10, 8, 6)
    meta_activation: str = "relu"
    meta_solver: str = "adam"
    meta_learning_rate: float = 0.001


class MetaClassifier(nn.Module):
    """Meta-Classifier that combines base model outputs.

    The Meta-Classifier uses separate MLP regressors for each
    forecast day, allowing the model to learn day-specific
    combination weights for the base model predictions.

    Attributes:
        config: EnsembleConfig with hyperparameters.
        lstm: LSTM base model.
        cnn: CNN base model.
        meta_mlps: List of MLP heads, one per forecast day.

    Example:
        >>> config = EnsembleConfig(output_horizon=7)
        >>> model = MetaClassifier(config)
        >>> predictions = model(x, features)  # Shape: (batch, 7)
    """

    def __init__(self, config: EnsembleConfig) -> None:
        """Initialize the Meta-Classifier.

        Args:
            config: Ensemble configuration.
        """
        super().__init__()
        self.config = config

        # Create base model configs
        lstm_config = DeepModelConfig(
            input_window=config.input_window,
            output_horizon=config.output_horizon,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
            n_temporal_features=config.n_temporal_features,
        )

        cnn_config = DeepModelConfig(
            input_window=config.input_window,
            output_horizon=config.output_horizon,
            hidden_size=config.lstm_hidden,
            dropout=config.dropout,
            n_temporal_features=config.n_temporal_features,
        )

        # Create base models
        self.lstm = LSTMModel(lstm_config)
        self.cnn = CNNModel(cnn_config)

        # Compute combined feature size
        # LSTM outputs: hidden_size
        # CNN outputs: num_filters[-1] (128 by default)
        lstm_output_size = config.lstm_hidden
        cnn_output_size = config.cnn_filters[-1]
        combined_size = lstm_output_size + cnn_output_size

        # Create separate MLP head for each forecast day
        self.meta_mlps = nn.ModuleList()
        for day in range(config.output_horizon):
            mlp = self._create_mlp(
                input_size=combined_size,
                hidden_layers=config.meta_hidden_layers,
                output_size=1,
                dropout=config.dropout,
            )
            self.meta_mlps.append(mlp)

        logger.info(
            f"Created MetaClassifier: LSTM({config.lstm_hidden}x{config.lstm_layers}) + "
            f"CNN({config.cnn_filters}) -> {config.output_horizon} MLPs"
        )

    def _create_mlp(
        self,
        input_size: int,
        hidden_layers: Tuple[int, ...],
        output_size: int,
        dropout: float,
    ) -> nn.Sequential:
        """Create an MLP with specified architecture.

        Args:
            input_size: Input feature dimension.
            hidden_layers: Tuple of hidden layer sizes.
            output_size: Output dimension.
            dropout: Dropout probability.

        Returns:
            Sequential MLP module.
        """
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the ensemble.

        Args:
            x: Input sequence of shape (batch, seq_len).
            features: Optional temporal features.

        Returns:
            Predictions of shape (batch, output_horizon).
        """
        # Get intermediate outputs from base models
        lstm_out = self.lstm.get_intermediate_output(x, features)
        cnn_out = self.cnn.get_intermediate_output(x, features)

        # Concatenate base model outputs
        combined = torch.cat([lstm_out, cnn_out], dim=-1)

        # Apply day-specific MLPs
        predictions = []
        for day_mlp in self.meta_mlps:
            day_pred = day_mlp(combined)
            predictions.append(day_pred)

        # Stack predictions: (batch, output_horizon)
        output = torch.cat(predictions, dim=-1)

        return output

    def get_base_predictions(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions from base models directly.

        Useful for comparing ensemble vs individual models.

        Args:
            x: Input sequence.
            features: Optional temporal features.

        Returns:
            Tuple of (lstm_predictions, cnn_predictions).
        """
        lstm_pred = self.lstm(x, features)
        cnn_pred = self.cnn(x, features)
        return lstm_pred, cnn_pred


class EnsembleForecaster:
    """High-level wrapper for ensemble forecasting with sklearn meta-learners.

    This class provides an alternative implementation using sklearn's
    MLPRegressor for the meta-learning step, with hyperparameter tuning
    via grid search.

    The workflow:
    1. Train base models (LSTM, CNN) using PyTorch
    2. Extract intermediate representations
    3. Train sklearn MLPRegressor for each forecast day
    4. Combine for final predictions

    Attributes:
        config: EnsembleConfig with hyperparameters.
        lstm_model: Trained LSTM base model.
        cnn_model: Trained CNN base model.
        meta_regressors: List of trained MLPRegressors.

    Example:
        >>> forecaster = EnsembleForecaster(config)
        >>> forecaster.fit(train_loader, val_loader)
        >>> predictions = forecaster.predict(test_data)
    """

    # Hyperparameter grid for meta-regressor tuning
    MLP_PARAM_GRID: Dict[str, List[Any]] = {
        "hidden_layer_sizes": [
            (10, 8, 6),
            (10, 6, 6),
            (10, 6, 4),
            (8, 8, 4),
            (8, 8, 6),
            (8, 6, 4),
            (8, 6, 6),
            (6, 6, 6),
            (6, 6, 4),
            (6, 4, 4),
        ],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "learning_rate_init": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        "learning_rate": ["adaptive", "constant"],
    }

    def __init__(self, config: EnsembleConfig) -> None:
        """Initialize the ensemble forecaster.

        Args:
            config: Ensemble configuration.
        """
        self.config = config
        self.lstm_model: Optional[LSTMModel] = None
        self.cnn_model: Optional[CNNModel] = None
        self.meta_regressors: List[MLPRegressor] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_base_models(self) -> None:
        """Create and initialize base models."""
        lstm_config = DeepModelConfig(
            input_window=self.config.input_window,
            output_horizon=self.config.output_horizon,
            hidden_size=self.config.lstm_hidden,
            num_layers=self.config.lstm_layers,
            dropout=self.config.dropout,
            n_temporal_features=self.config.n_temporal_features,
        )

        cnn_config = DeepModelConfig(
            input_window=self.config.input_window,
            output_horizon=self.config.output_horizon,
            hidden_size=self.config.lstm_hidden,
            dropout=self.config.dropout,
            n_temporal_features=self.config.n_temporal_features,
        )

        self.lstm_model = LSTMModel(lstm_config).to(self.device)
        self.cnn_model = CNNModel(cnn_config).to(self.device)

    def extract_base_features(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Extract combined features from base models.

        Args:
            x: Input sequence tensor.
            features: Optional temporal features.

        Returns:
            Combined feature array of shape (batch, combined_dim).
        """
        self.lstm_model.eval()
        self.cnn_model.eval()

        with torch.no_grad():
            x = x.to(self.device)
            if features is not None:
                features = features.to(self.device)

            lstm_feat = self.lstm_model.get_intermediate_output(x, features)
            cnn_feat = self.cnn_model.get_intermediate_output(x, features)

            combined = torch.cat([lstm_feat, cnn_feat], dim=-1)

        return combined.cpu().numpy()

    def train_meta_regressors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune_hyperparams: bool = False,
    ) -> None:
        """Train meta-regressors for each forecast day.

        Args:
            X: Combined features from base models.
            y: Target values of shape (n_samples, output_horizon).
            tune_hyperparams: Whether to perform grid search.
        """
        self.meta_regressors = []

        for day in range(self.config.output_horizon):
            logger.info(f"Training meta-regressor for day {day + 1}...")

            y_day = y[:, day]

            if tune_hyperparams:
                # Grid search for best hyperparameters
                base_mlp = MLPRegressor(max_iter=500, early_stopping=True)
                grid_search = GridSearchCV(
                    base_mlp,
                    self.MLP_PARAM_GRID,
                    cv=3,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
                grid_search.fit(X, y_day)
                regressor = grid_search.best_estimator_
                logger.info(f"Day {day + 1} best params: {grid_search.best_params_}")
            else:
                # Use default configuration
                regressor = MLPRegressor(
                    hidden_layer_sizes=self.config.meta_hidden_layers,
                    activation=self.config.meta_activation,
                    solver=self.config.meta_solver,
                    learning_rate_init=self.config.meta_learning_rate,
                    max_iter=500,
                    early_stopping=True,
                )
                regressor.fit(X, y_day)

            self.meta_regressors.append(regressor)

    def predict(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Generate ensemble predictions.

        Args:
            x: Input sequence tensor.
            features: Optional temporal features.

        Returns:
            Predictions of shape (batch, output_horizon).
        """
        # Extract features from base models
        combined_features = self.extract_base_features(x, features)

        # Get predictions from each meta-regressor
        predictions = []
        for regressor in self.meta_regressors:
            day_pred = regressor.predict(combined_features)
            predictions.append(day_pred)

        return np.column_stack(predictions)


class SimpleEnsemble(nn.Module):
    """Simple averaging ensemble of multiple models.

    This ensemble simply averages predictions from all base models,
    serving as a baseline for the more complex Meta-Classifier.

    Example:
        >>> ensemble = SimpleEnsemble([lstm_model, cnn_model])
        >>> predictions = ensemble(x, features)
    """

    def __init__(self, models: List[nn.Module]) -> None:
        """Initialize the simple ensemble.

        Args:
            models: List of base models.
        """
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass averaging all model predictions.

        Args:
            x: Input sequence.
            features: Optional temporal features.

        Returns:
            Averaged predictions.
        """
        predictions = []
        for model in self.models:
            pred = model(x, features)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)

