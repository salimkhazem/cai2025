"""Base classes for data loading and dataset handling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        input_window: Number of days used as input sequence (lookback window).
        output_horizon: Number of days to predict (forecast horizon).
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
        normalize: Whether to apply Min-Max normalization.
        aggregation: Time aggregation level ('daily', 'hourly').
    """

    input_window: int = 120
    output_horizon: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    normalize: bool = True
    aggregation: str = "daily"


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting with sliding window.

    This dataset creates input-output pairs using a sliding window approach,
    where each sample consists of:
    - Input: `input_window` consecutive time steps
    - Output: `output_horizon` consecutive time steps following the input

    Attributes:
        data: Normalized time series data as numpy array.
        features: Optional temporal features as numpy array.
        input_window: Number of time steps in input sequence.
        output_horizon: Number of time steps to predict.
        use_features: Whether to include temporal features.

    Example:
        >>> dataset = TimeSeriesDataset(
        ...     data=consumption_data,
        ...     features=temporal_features,
        ...     input_window=120,
        ...     output_horizon=7
        ... )
        >>> x, features, y = dataset[0]
    """

    def __init__(
        self,
        data: np.ndarray,
        features: Optional[np.ndarray] = None,
        input_window: int = 120,
        output_horizon: int = 7,
    ) -> None:
        """Initialize the TimeSeriesDataset.

        Args:
            data: Time series data of shape (n_timesteps,) or (n_timesteps, n_features).
            features: Optional temporal features of shape (n_timesteps, n_temporal_features).
            input_window: Number of time steps in input sequence.
            output_horizon: Number of time steps to predict.

        Raises:
            ValueError: If data length is insufficient for the given windows.
        """
        self.data = data.astype(np.float32)
        self.features = features.astype(np.float32) if features is not None else None
        self.input_window = input_window
        self.output_horizon = output_horizon
        self.use_features = features is not None

        min_length = input_window + output_horizon
        if len(data) < min_length:
            raise ValueError(
                f"Data length ({len(data)}) must be at least "
                f"input_window + output_horizon ({min_length})"
            )

        self.n_samples = len(data) - input_window - output_horizon + 1
        logger.info(f"Created dataset with {self.n_samples} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple containing:
                - x: Input sequence tensor of shape (input_window,)
                - features: Temporal features tensor (only if use_features=True)
                - y: Target sequence tensor of shape (output_horizon,)
        """
        x_start = idx
        x_end = idx + self.input_window
        y_start = x_end
        y_end = y_start + self.output_horizon

        x = torch.from_numpy(self.data[x_start:x_end])
        y = torch.from_numpy(self.data[y_start:y_end])

        if self.use_features:
            # Include features for both input and output periods
            feat = torch.from_numpy(self.features[x_start:y_end])
            return x, feat, y
        else:
            # Return only x and y when no features
            return x, y


class BaseDataLoader(ABC):
    """Abstract base class for data loaders.

    This class defines the interface for loading and preprocessing
    energy consumption datasets.

    Attributes:
        config: DataConfig instance with loading parameters.
        scaler: MinMaxScaler for data normalization.
        data: Loaded and preprocessed DataFrame.
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        """Initialize the data loader.

        Args:
            config: Configuration for data loading. Uses defaults if None.
        """
        self.config = config or DataConfig()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data: Optional[pd.DataFrame] = None
        self._raw_data: Optional[pd.DataFrame] = None

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load data from the specified path.

        Args:
            path: Path to the data file.

        Returns:
            DataFrame with loaded data.
        """
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the loaded data.

        Args:
            df: Raw DataFrame.

        Returns:
            Preprocessed DataFrame with datetime index and consumption values.
        """
        pass

    def get_consumption_series(self) -> np.ndarray:
        """Get the consumption time series as numpy array.

        Returns:
            1D numpy array of consumption values.

        Raises:
            ValueError: If data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data["consumption"].values

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using Min-Max scaling.

        Args:
            data: Data to normalize.

        Returns:
            Normalized data in range [0, 1].
        """
        return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """Inverse the normalization.

        Args:
            data: Normalized data.

        Returns:
            Data in original scale.
        """
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def create_datasets(
        self,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """Create train, validation, and test datasets.

        Args:
            features: Optional temporal features array.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            ValueError: If data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        consumption = self.get_consumption_series()

        if self.config.normalize:
            consumption = self.normalize(consumption)

        n = len(consumption)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        # Split data
        train_data = consumption[:train_end]
        val_data = consumption[train_end - self.config.input_window : val_end]
        test_data = consumption[val_end - self.config.input_window :]

        # Split features if provided
        train_feat = features[:train_end] if features is not None else None
        val_feat = (
            features[train_end - self.config.input_window : val_end]
            if features is not None
            else None
        )
        test_feat = (
            features[val_end - self.config.input_window :]
            if features is not None
            else None
        )

        train_dataset = TimeSeriesDataset(
            train_data,
            train_feat,
            self.config.input_window,
            self.config.output_horizon,
        )
        val_dataset = TimeSeriesDataset(
            val_data,
            val_feat,
            self.config.input_window,
            self.config.output_horizon,
        )
        test_dataset = TimeSeriesDataset(
            test_data,
            test_feat,
            self.config.input_window,
            self.config.output_horizon,
        )

        logger.info(
            f"Created datasets - Train: {len(train_dataset)}, "
            f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get the date range of the loaded data.

        Returns:
            Tuple of (start_date, end_date).

        Raises:
            ValueError: If data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data.index.min(), self.data.index.max()

    def get_statistics(self) -> dict:
        """Get summary statistics of the loaded data.

        Returns:
            Dictionary with statistics (mean, std, min, max, etc.).

        Raises:
            ValueError: If data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        consumption = self.data["consumption"]
        return {
            "mean": consumption.mean(),
            "std": consumption.std(),
            "min": consumption.min(),
            "max": consumption.max(),
            "n_samples": len(consumption),
            "date_range": self.get_date_range(),
        }

