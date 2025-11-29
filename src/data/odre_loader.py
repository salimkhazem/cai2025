"""Data loader for the ODRE (Open Data Réseaux Énergies) French dataset.

The ODRE dataset contains regional electricity and gas consumption data
for France, collected at 30-minute intervals.

Reference:
    Open Data Réseaux Énergies - https://opendata.reseaux-energies.fr/
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from .base import BaseDataLoader, DataConfig

logger = logging.getLogger(__name__)


# French region codes and names
REGION_CODES = {
    11: "Île-de-France",
    24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté",
    28: "Normandie",
    32: "Hauts-de-France",
    44: "Grand Est",
    52: "Pays de la Loire",
    53: "Bretagne",
    75: "Nouvelle-Aquitaine",
    76: "Occitanie",
    84: "Auvergne-Rhône-Alpes",
    93: "Provence-Alpes-Côte d'Azur",
}


class ODREDataLoader(BaseDataLoader):
    """Data loader for the ODRE French regional electricity dataset.

    This loader handles the Open Data Réseaux Énergies dataset which contains
    French regional electricity consumption at 30-minute intervals.

    Attributes:
        region_codes: List of region codes to load. If None, loads national total.
        config: DataConfig instance with loading parameters.

    Example:
        >>> loader = ODREDataLoader(region_codes=[11, 75])  # IDF + Nouvelle-Aquitaine
        >>> loader.load("input/odre_data/regional_daily_brut_consumption.csv")
        >>> train, val, test = loader.create_datasets()
    """

    def __init__(
        self,
        region_codes: Optional[List[int]] = None,
        config: Optional[DataConfig] = None,
    ) -> None:
        """Initialize the ODRE data loader.

        Args:
            region_codes: List of region codes to load (see REGION_CODES).
                         If None, aggregates across all regions (national).
            config: Configuration for data loading.
        """
        super().__init__(config)
        self.region_codes = region_codes

    def load(self, path: str) -> pd.DataFrame:
        """Load the ODRE regional consumption dataset.

        Args:
            path: Path to the CSV file.

        Returns:
            DataFrame with datetime index and consumption columns.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info(f"Loading ODRE dataset from {path}")

        # Read with French format (semicolon separator)
        df = pd.read_csv(
            path,
            sep=";",
            low_memory=False,
        )

        logger.info(f"Loaded {len(df)} records")

        self._raw_data = df
        self.data = self.preprocess(df)

        return self.data

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the ODRE dataset.

        Preprocessing steps:
        1. Parse datetime from Date and Heure columns
        2. Select electricity consumption column
        3. Filter by region if specified
        4. Aggregate to daily if configured
        5. Handle missing values

        Args:
            df: Raw DataFrame from load().

        Returns:
            Preprocessed DataFrame with datetime index and 'consumption' column.
        """
        logger.info("Preprocessing ODRE dataset...")

        # Create datetime from Date and Heure columns
        df["datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Heure"], format="%Y-%m-%d %H:%M"
        )

        # Select electricity consumption column (RTE data)
        consumption_col = "Consommation brute électricité (MW) - RTE"

        if consumption_col not in df.columns:
            raise ValueError(f"Column '{consumption_col}' not found in dataset")

        # Filter by region if specified
        if self.region_codes is not None:
            df = df[df["Code INSEE région"].isin(self.region_codes)]
            logger.info(f"Filtered to {len(self.region_codes)} regions")

        # Convert consumption to numeric, coercing errors to NaN
        df["consumption"] = pd.to_numeric(df[consumption_col], errors="coerce")

        # Group by datetime and sum consumption
        result = df.groupby("datetime")["consumption"].sum().to_frame()
        result = result.sort_index()

        # Aggregate to daily if configured
        if self.config.aggregation == "daily":
            result = result.resample("D").sum()
            logger.info(f"Aggregated to daily: {len(result)} days")

        # Handle missing values
        # Use forward-fill for small gaps, then backward-fill for edges
        result["consumption"] = result["consumption"].ffill().bfill()

        # Remove any remaining NaN or zero values
        result = result[result["consumption"] > 0]

        logger.info(
            f"Preprocessing complete: {len(result)} samples, "
            f"range [{result.index.min()} to {result.index.max()}]"
        )

        return result

    def get_available_regions(self) -> dict:
        """Get dictionary of available region codes and names.

        Returns:
            Dictionary mapping region codes to region names.
        """
        return REGION_CODES.copy()

    def get_region_statistics(self) -> pd.DataFrame:
        """Get statistics for each region.

        Returns:
            DataFrame with mean, std, count for each region.

        Raises:
            ValueError: If data hasn't been loaded yet.
        """
        if self._raw_data is None:
            raise ValueError("Load data first to get region statistics.")

        consumption_col = "Consommation brute électricité (MW) - RTE"

        # Convert to numeric
        data = self._raw_data.copy()
        data["consumption"] = pd.to_numeric(data[consumption_col], errors="coerce")

        stats = (
            data.groupby("Code INSEE région")["consumption"]
            .agg(["mean", "std", "count", "sum"])
            .round(2)
        )
        stats["region_name"] = stats.index.map(REGION_CODES)

        return stats

