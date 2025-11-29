"""Baseline forecasting models for comparison.

This module implements traditional and simple baseline models:
- Naive: Repeats the last observed value
- Moving Average: Uses rolling mean of recent values
- ARIMA: Autoregressive Integrated Moving Average

These baselines are essential for demonstrating the value of
deep learning approaches over traditional methods.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for baseline models.

    Attributes:
        input_window: Number of past observations to use.
        output_horizon: Number of future steps to predict.
    """

    input_window: int = 120
    output_horizon: int = 7


class BaseModel(ABC):
    """Abstract base class for all forecasting models.

    All models must implement:
        - fit: Train the model on historical data
        - predict: Generate forecasts for future horizons
    """

    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        """Initialize the model.

        Args:
            config: Model configuration.
        """
        self.config = config or BaselineConfig()
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: np.ndarray) -> "BaseModel":
        """Fit the model on training data.

        Args:
            train_data: 1D array of historical values.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, history: np.ndarray) -> np.ndarray:
        """Generate predictions for the forecast horizon.

        Args:
            history: Recent history (input_window values).

        Returns:
            Array of predictions (output_horizon values).
        """
        pass

    def evaluate(
        self, test_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model using sliding window on test data.

        Args:
            test_data: Test data array.

        Returns:
            Tuple of (predictions, actuals, errors) arrays.
        """
        n_samples = len(test_data) - self.config.input_window - self.config.output_horizon + 1

        all_preds = []
        all_actuals = []

        for i in range(n_samples):
            history = test_data[i : i + self.config.input_window]
            actual = test_data[
                i + self.config.input_window : i
                + self.config.input_window
                + self.config.output_horizon
            ]

            pred = self.predict(history)
            all_preds.append(pred)
            all_actuals.append(actual)

        predictions = np.array(all_preds)
        actuals = np.array(all_actuals)
        errors = predictions - actuals

        return predictions, actuals, errors


class NaiveModel(BaseModel):
    """Naive baseline that predicts the last observed value.

    This is the simplest possible baseline - it assumes the future
    will be exactly like the most recent observation. Any model
    worth using should significantly outperform this baseline.

    Example:
        >>> model = NaiveModel()
        >>> model.fit(train_data)
        >>> preds = model.predict(history)
    """

    def fit(self, train_data: np.ndarray) -> "NaiveModel":
        """Naive model doesn't need fitting.

        Args:
            train_data: Training data (ignored).

        Returns:
            Self for method chaining.
        """
        self.is_fitted = True
        logger.info("Naive model fitted (no parameters to learn)")
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict the last value for all future steps.

        Args:
            history: Recent history array.

        Returns:
            Array repeating the last observed value.
        """
        last_value = history[-1]
        return np.full(self.config.output_horizon, last_value)


class MovingAverageModel(BaseModel):
    """Moving average baseline model.

    Predicts the mean of the last `window_size` observations
    for all future time steps.

    Attributes:
        window_size: Number of recent values to average.

    Example:
        >>> model = MovingAverageModel(window_size=7)
        >>> model.fit(train_data)
        >>> preds = model.predict(history)
    """

    def __init__(
        self,
        window_size: int = 7,
        config: Optional[BaselineConfig] = None,
    ) -> None:
        """Initialize the moving average model.

        Args:
            window_size: Number of recent values to average.
            config: Model configuration.
        """
        super().__init__(config)
        self.window_size = window_size

    def fit(self, train_data: np.ndarray) -> "MovingAverageModel":
        """Moving average doesn't need fitting.

        Args:
            train_data: Training data (ignored).

        Returns:
            Self for method chaining.
        """
        self.is_fitted = True
        logger.info(f"Moving average model fitted (window_size={self.window_size})")
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict the moving average for all future steps.

        Args:
            history: Recent history array.

        Returns:
            Array with moving average value.
        """
        ma_value = np.mean(history[-self.window_size :])
        return np.full(self.config.output_horizon, ma_value)


class ARIMAModel(BaseModel):
    """ARIMA (AutoRegressive Integrated Moving Average) model.

    ARIMA is a classical time series forecasting method that combines:
    - AR (AutoRegressive): Regression on past values
    - I (Integrated): Differencing to achieve stationarity
    - MA (Moving Average): Regression on past forecast errors

    Attributes:
        order: Tuple (p, d, q) for ARIMA parameters.
        seasonal_order: Optional tuple for seasonal ARIMA.

    Example:
        >>> model = ARIMAModel(order=(5, 1, 0))
        >>> model.fit(train_data)
        >>> preds = model.predict(history)
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (5, 1, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        config: Optional[BaselineConfig] = None,
    ) -> None:
        """Initialize the ARIMA model.

        Args:
            order: (p, d, q) parameters for ARIMA.
            seasonal_order: (P, D, Q, s) for seasonal ARIMA.
            config: Model configuration.
        """
        super().__init__(config)
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted_model = None

    def fit(self, train_data: np.ndarray) -> "ARIMAModel":
        """Fit the ARIMA model on training data.

        Args:
            train_data: 1D array of historical values.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting ARIMA{self.order} model...")

        try:
            model = ARIMA(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
            )
            self._fitted_model = model.fit()
            self.is_fitted = True
            logger.info("ARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise

        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Generate multi-step forecasts.

        For each prediction, we refit ARIMA on the provided history
        to generate the forecast. This is computationally expensive
        but provides better results for multi-step forecasting.

        Args:
            history: Recent history (input_window values).

        Returns:
            Array of predictions (output_horizon values).
        """
        try:
            # Refit on the provided history for better predictions
            model = ARIMA(history, order=self.order)
            fitted = model.fit()

            # Forecast the horizon
            forecast = fitted.forecast(steps=self.config.output_horizon)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"ARIMA prediction failed: {e}, using naive fallback")
            return np.full(self.config.output_horizon, history[-1])


class ExponentialSmoothingModel(BaseModel):
    """Exponential Smoothing (Holt-Winters) model.

    This model captures trend and seasonality using exponential
    smoothing techniques.

    Attributes:
        trend: Type of trend ('add', 'mul', None).
        seasonal: Type of seasonality ('add', 'mul', None).
        seasonal_periods: Number of periods in a season.

    Example:
        >>> model = ExponentialSmoothingModel(seasonal_periods=7)
        >>> model.fit(train_data)
        >>> preds = model.predict(history)
    """

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: int = 7,
        config: Optional[BaselineConfig] = None,
    ) -> None:
        """Initialize the Exponential Smoothing model.

        Args:
            trend: Type of trend component.
            seasonal: Type of seasonal component.
            seasonal_periods: Length of the seasonal cycle.
            config: Model configuration.
        """
        super().__init__(config)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def fit(self, train_data: np.ndarray) -> "ExponentialSmoothingModel":
        """Fit the Exponential Smoothing model.

        Args:
            train_data: 1D array of historical values.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting Exponential Smoothing model...")
        self.is_fitted = True
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Generate forecasts using Exponential Smoothing.

        Args:
            history: Recent history array.

        Returns:
            Array of predictions.
        """
        try:
            # Need at least 2 seasonal cycles for seasonal models
            if len(history) < 2 * self.seasonal_periods:
                # Fall back to simple exponential smoothing
                model = ExponentialSmoothing(
                    history,
                    trend=self.trend,
                    seasonal=None,
                )
            else:
                model = ExponentialSmoothing(
                    history,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                )

            fitted = model.fit()
            forecast = fitted.forecast(self.config.output_horizon)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"ExpSmoothing failed: {e}, using MA fallback")
            return np.full(self.config.output_horizon, np.mean(history[-7:]))


class SeasonalNaiveModel(BaseModel):
    """Seasonal naive baseline.

    Predicts by repeating the values from the same period
    in the previous seasonal cycle.

    Attributes:
        seasonal_period: Length of the seasonal cycle (e.g., 7 for weekly).

    Example:
        >>> model = SeasonalNaiveModel(seasonal_period=7)
        >>> model.fit(train_data)
        >>> preds = model.predict(history)  # Predicts week-ago values
    """

    def __init__(
        self,
        seasonal_period: int = 7,
        config: Optional[BaselineConfig] = None,
    ) -> None:
        """Initialize the Seasonal Naive model.

        Args:
            seasonal_period: Length of the seasonal cycle.
            config: Model configuration.
        """
        super().__init__(config)
        self.seasonal_period = seasonal_period

    def fit(self, train_data: np.ndarray) -> "SeasonalNaiveModel":
        """Seasonal naive doesn't need fitting.

        Args:
            train_data: Training data (ignored).

        Returns:
            Self for method chaining.
        """
        self.is_fitted = True
        logger.info(
            f"Seasonal naive model fitted (period={self.seasonal_period})"
        )
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict using values from the previous seasonal cycle.

        Args:
            history: Recent history array.

        Returns:
            Array of predictions.
        """
        predictions = []
        for i in range(self.config.output_horizon):
            # Get the value from one seasonal period ago
            idx = len(history) - self.seasonal_period + i
            if idx >= 0:
                predictions.append(history[idx])
            else:
                predictions.append(history[-1])

        return np.array(predictions)

