"""Data loader for the UCI Electricity Load Diagrams dataset.

The UCI Electricity dataset contains electricity consumption data
from 370 clients collected every 15 minutes from 2011 to 2014.

Reference:
    UCI Machine Learning Repository
    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from .base import BaseDataLoader, DataConfig

logger = logging.getLogger(__name__)


class UCLDataLoader(BaseDataLoader):
    """Data loader for the UCI Electricity Load Diagrams dataset.

    This loader handles the UCI dataset which contains electricity consumption
    measurements from 370 Portuguese clients at 15-minute intervals from
    2011 to 2014 (approximately 140,000 timestamps per client).

    Attributes:
        client_ids: List of client IDs to load. If None, loads aggregated data.
        config: DataConfig instance with loading parameters.

    Example:
        >>> loader = UCLDataLoader(client_ids=["MT_001", "MT_002"])
        >>> loader.load("input/UCL_dataset/LD2011_2014.txt")
        >>> train, val, test = loader.create_datasets()
    """

    def __init__(
        self,
        client_ids: Optional[List[str]] = None,
        config: Optional[DataConfig] = None,
    ) -> None:
        """Initialize the UCL data loader.

        Args:
            client_ids: List of client IDs to load (e.g., ["MT_001", "MT_002"]).
                       If None, aggregates consumption across all clients.
            config: Configuration for data loading.
        """
        super().__init__(config)
        self.client_ids = client_ids

    def load(self, path: str) -> pd.DataFrame:
        """Load the UCI Electricity dataset.

        Args:
            path: Path to the LD2011_2014.txt file.

        Returns:
            DataFrame with datetime index and consumption columns.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If specified client_ids are not in the dataset.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info(f"Loading UCI Electricity dataset from {path}")

        # Read the dataset with European format (semicolon separator, comma decimal)
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            parse_dates=[0],
            index_col=0,
            low_memory=False,
        )

        # Clean column names (remove quotes if present)
        df.columns = [col.strip('"') for col in df.columns]
        df.index.name = "datetime"

        logger.info(f"Loaded {len(df)} timestamps for {len(df.columns)} clients")

        self._raw_data = df
        self.data = self.preprocess(df)

        return self.data

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the UCI dataset.

        Preprocessing steps:
        1. Select specified clients or aggregate all
        2. Aggregate to daily consumption if configured
        3. Handle missing values
        4. Create final consumption series

        Args:
            df: Raw DataFrame from load().

        Returns:
            Preprocessed DataFrame with datetime index and 'consumption' column.
        """
        logger.info("Preprocessing UCI dataset...")

        # Select clients
        if self.client_ids is not None:
            missing = set(self.client_ids) - set(df.columns)
            if missing:
                raise ValueError(f"Client IDs not found in dataset: {missing}")
            consumption = df[self.client_ids].sum(axis=1)
            logger.info(f"Selected {len(self.client_ids)} clients")
        else:
            # Aggregate all clients
            consumption = df.sum(axis=1)
            logger.info("Aggregating consumption across all clients")

        # Convert to DataFrame
        result = pd.DataFrame({"consumption": consumption})

        # Handle missing/zero values (some clients start recording later)
        # Replace zeros with NaN, then forward-fill
        result["consumption"] = result["consumption"].replace(0, np.nan)
        
        # For early timestamps with many inactive clients, use backward fill first
        result["consumption"] = result["consumption"].bfill().ffill()

        # Aggregate to daily if configured
        if self.config.aggregation == "daily":
            result = result.resample("D").sum()
            logger.info(f"Aggregated to daily: {len(result)} days")

        # Remove any remaining NaN values
        result = result.dropna()

        logger.info(
            f"Preprocessing complete: {len(result)} samples, "
            f"range [{result.index.min()} to {result.index.max()}]"
        )

        return result

    def get_available_clients(self) -> List[str]:
        """Get list of available client IDs.

        Returns:
            List of client ID strings.

        Raises:
            ValueError: If data hasn't been loaded yet.
        """
        if self._raw_data is None:
            raise ValueError("Load data first to get available clients.")
        return list(self._raw_data.columns)

    def get_client_statistics(self) -> pd.DataFrame:
        """Get statistics for each client.

        Returns:
            DataFrame with mean, std, min, max for each client.

        Raises:
            ValueError: If data hasn't been loaded yet.
        """
        if self._raw_data is None:
            raise ValueError("Load data first to get client statistics.")

        stats = self._raw_data.describe().T
        stats["total"] = self._raw_data.sum()
        stats["active_ratio"] = (self._raw_data > 0).mean()

        return stats

