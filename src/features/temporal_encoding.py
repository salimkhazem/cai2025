"""Cyclical temporal encoding for time series forecasting.

This module implements sine/cosine transformations for encoding periodic
temporal features, preserving the cyclical nature of time variables.

The key insight is that raw temporal features (e.g., day-of-week 0-6) have
artificial discontinuities (Sunday=0 is far from Saturday=6 numerically,
but adjacent temporally). Cyclical encoding maps these to a continuous
circular space where the distance reflects true temporal proximity.

Reference:
    - Bandara et al. (2020): "Forecasting across time series databases
      using recurrent neural networks on groups of similar series"
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CyclicalEncodingConfig:
    """Configuration for cyclical encoding.

    Attributes:
        features: List of features to encode cyclically.
        include_raw: Whether to also include raw (non-encoded) features.
    """

    features: List[str] = field(
        default_factory=lambda: [
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month_of_year",
        ]
    )
    include_raw: bool = False


class CyclicalEncoder:
    """Encodes periodic features using sine and cosine transformations.

    For a feature with maximum value `max_val`, the encoding is:
        - sin_feature = sin(2π × value / max_val)
        - cos_feature = cos(2π × value / max_val)

    This maps the periodic feature to a unit circle, where:
        - Adjacent values are close in the encoded space
        - The transition from max to 0 is smooth (no discontinuity)

    Example:
        >>> encoder = CyclicalEncoder()
        >>> day_of_week = np.array([0, 1, 2, 3, 4, 5, 6])
        >>> sin_enc, cos_enc = encoder.encode(day_of_week, max_val=7)
        >>> # Now Sunday (0) is close to Saturday (6) in encoded space
    """

    def encode(
        self, values: np.ndarray, max_val: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode values using sine and cosine transformation.

        Args:
            values: Array of periodic values to encode.
            max_val: The period of the cycle (e.g., 7 for day-of-week).

        Returns:
            Tuple of (sin_encoded, cos_encoded) arrays.
        """
        angle = 2 * np.pi * values / max_val
        return np.sin(angle), np.cos(angle)

    def encode_dataframe(
        self, df: pd.DataFrame, column: str, max_val: float
    ) -> pd.DataFrame:
        """Encode a DataFrame column with cyclical encoding.

        Args:
            df: Input DataFrame.
            column: Name of the column to encode.
            max_val: The period of the cycle.

        Returns:
            DataFrame with added sin_{column} and cos_{column} columns.
        """
        values = df[column].values
        sin_vals, cos_vals = self.encode(values, max_val)
        df[f"sin_{column}"] = sin_vals
        df[f"cos_{column}"] = cos_vals
        return df


class CalendarFeatureExtractor:
    """Extracts calendar-based features from datetime index.

    Features extracted:
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - day_of_month: Day of month (1-31)
        - day_of_year: Day of year (1-366)
        - week_of_year: Week of year (1-53)
        - month_of_year: Month (1-12)
        - quarter: Quarter (1-4)
        - is_weekend: Binary weekend indicator
        - is_business_day: Binary business day indicator

    Example:
        >>> extractor = CalendarFeatureExtractor()
        >>> dates = pd.date_range("2023-01-01", periods=365, freq="D")
        >>> features = extractor.extract(dates)
    """

    # Maximum values for cyclical encoding
    MAX_VALUES: Dict[str, float] = {
        "day_of_week": 7,
        "day_of_month": 31,
        "day_of_year": 366,
        "week_of_year": 53,
        "month_of_year": 12,
        "quarter": 4,
        "hour": 24,
        "minute": 60,
    }

    def extract(self, datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Extract calendar features from datetime index.

        Args:
            datetime_index: Pandas DatetimeIndex.

        Returns:
            DataFrame with extracted calendar features.
        """
        features = pd.DataFrame(index=datetime_index)

        # Basic temporal features
        features["day_of_week"] = datetime_index.dayofweek
        features["day_of_month"] = datetime_index.day
        features["day_of_year"] = datetime_index.dayofyear
        features["week_of_year"] = datetime_index.isocalendar().week.values
        features["month_of_year"] = datetime_index.month
        features["quarter"] = datetime_index.quarter

        # Binary features
        features["is_weekend"] = (datetime_index.dayofweek >= 5).astype(int)
        features["is_business_day"] = (
            pd.to_datetime(datetime_index).map(lambda x: x.weekday() < 5)
        ).astype(int)

        # Hour and minute if intraday data
        if hasattr(datetime_index, "hour"):
            features["hour"] = datetime_index.hour
            features["minute"] = datetime_index.minute

        logger.info(f"Extracted {len(features.columns)} calendar features")

        return features

    def get_feature_correlations(
        self, features: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:
        """Compute Pearson correlation between features and target.

        Args:
            features: DataFrame of calendar features.
            target: Series of target values (e.g., consumption).

        Returns:
            DataFrame with correlation coefficients for each feature.
        """
        correlations = {}
        for col in features.columns:
            correlations[col] = features[col].corr(target)

        result = pd.DataFrame(
            {"feature": list(correlations.keys()), "correlation": list(correlations.values())}
        )
        result["abs_correlation"] = result["correlation"].abs()
        result = result.sort_values("abs_correlation", ascending=False)

        return result


class TemporalFeatureEngine:
    """Complete temporal feature engineering pipeline.

    This class combines calendar feature extraction and cyclical encoding
    to produce a comprehensive set of temporal features for forecasting.

    The pipeline:
    1. Extract raw calendar features from datetime index
    2. Apply cyclical encoding to periodic features
    3. Compute feature correlations with target (optional)
    4. Select top features based on correlation (optional)

    Attributes:
        config: CyclicalEncodingConfig for encoding settings.
        encoder: CyclicalEncoder instance.
        extractor: CalendarFeatureExtractor instance.
        feature_correlations: Stored correlations after fitting.

    Example:
        >>> engine = TemporalFeatureEngine()
        >>> features = engine.fit_transform(df.index, df["consumption"])
        >>> print(engine.feature_correlations)
    """

    def __init__(self, config: Optional[CyclicalEncodingConfig] = None) -> None:
        """Initialize the temporal feature engine.

        Args:
            config: Configuration for cyclical encoding.
        """
        self.config = config or CyclicalEncodingConfig()
        self.encoder = CyclicalEncoder()
        self.extractor = CalendarFeatureExtractor()
        self.feature_correlations: Optional[pd.DataFrame] = None

    def extract_features(self, datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Extract all temporal features.

        Args:
            datetime_index: Pandas DatetimeIndex.

        Returns:
            DataFrame with raw and cyclically encoded features.
        """
        # Extract raw calendar features
        features = self.extractor.extract(datetime_index)

        # Apply cyclical encoding to configured features
        for feature in self.config.features:
            if feature in features.columns:
                max_val = self.extractor.MAX_VALUES.get(feature, features[feature].max())
                features = self.encoder.encode_dataframe(features, feature, max_val)

        # Remove raw features if not needed
        if not self.config.include_raw:
            cols_to_drop = [
                col
                for col in self.config.features
                if col in features.columns and col not in ["is_weekend", "is_business_day"]
            ]
            features = features.drop(columns=cols_to_drop)

        logger.info(f"Generated {len(features.columns)} temporal features")

        return features

    def fit_transform(
        self,
        datetime_index: pd.DatetimeIndex,
        target: Optional[pd.Series] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Extract features and optionally compute correlations.

        Args:
            datetime_index: Pandas DatetimeIndex.
            target: Optional target series for correlation analysis.
            top_k: If set, return only top_k features by correlation.

        Returns:
            DataFrame with temporal features.
        """
        features = self.extract_features(datetime_index)

        if target is not None:
            # Align target with features
            aligned_target = target.reindex(datetime_index)
            self.feature_correlations = self.extractor.get_feature_correlations(
                features, aligned_target
            )
            logger.info("Computed feature correlations with target")

            # Select top features if requested
            if top_k is not None and self.feature_correlations is not None:
                top_features = self.feature_correlations.head(top_k)["feature"].tolist()
                features = features[top_features]
                logger.info(f"Selected top {top_k} features by correlation")

        return features

    def get_correlation_table(self) -> pd.DataFrame:
        """Get the correlation table from last fit_transform call.

        Returns:
            DataFrame with feature correlations.

        Raises:
            ValueError: If fit_transform hasn't been called with a target.
        """
        if self.feature_correlations is None:
            raise ValueError("Call fit_transform with target first.")
        return self.feature_correlations

    def transform(self, datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Transform new datetime index using the same features.

        Args:
            datetime_index: New DatetimeIndex to transform.

        Returns:
            DataFrame with temporal features.
        """
        return self.extract_features(datetime_index)


def compute_raw_vs_encoded_correlations(
    datetime_index: pd.DatetimeIndex, target: pd.Series
) -> pd.DataFrame:
    """Compare correlations of raw vs cyclically encoded features.

    This function demonstrates the benefit of cyclical encoding by
    comparing Pearson correlations of raw features against their
    sin/cos encoded versions.

    Args:
        datetime_index: Pandas DatetimeIndex.
        target: Target series (e.g., consumption).

    Returns:
        DataFrame comparing raw and encoded correlations.

    Example:
        >>> comparison = compute_raw_vs_encoded_correlations(df.index, df["consumption"])
        >>> print(comparison)
        # Shows that cos_day_of_year often has higher correlation than raw day_of_year
    """
    # Extract raw features
    extractor = CalendarFeatureExtractor()
    raw_features = extractor.extract(datetime_index)

    # Compute raw correlations
    raw_corr = {}
    for col in ["day_of_week", "day_of_month", "day_of_year", "week_of_year", "month_of_year"]:
        if col in raw_features.columns:
            raw_corr[col] = raw_features[col].corr(target)

    # Extract encoded features
    engine = TemporalFeatureEngine(CyclicalEncodingConfig(include_raw=False))
    encoded_features = engine.extract_features(datetime_index)

    # Compute encoded correlations
    encoded_corr = {}
    for col in encoded_features.columns:
        encoded_corr[col] = encoded_features[col].corr(target)

    # Build comparison table
    results = []
    for feature in raw_corr:
        sin_col = f"sin_{feature}"
        cos_col = f"cos_{feature}"

        sin_corr = encoded_corr.get(sin_col, np.nan)
        cos_corr = encoded_corr.get(cos_col, np.nan)
        best_encoded = max(abs(sin_corr), abs(cos_corr))

        results.append(
            {
                "feature": feature,
                "raw_correlation": raw_corr[feature],
                "sin_correlation": sin_corr,
                "cos_correlation": cos_corr,
                "best_encoded_abs": best_encoded,
                "improvement": best_encoded - abs(raw_corr[feature]),
            }
        )

    return pd.DataFrame(results)

