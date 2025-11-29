"""Training callbacks for early stopping and model checkpointing."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting.

    Monitors a metric and stops training when it stops improving.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        counter: Current number of epochs without improvement.
        best_score: Best observed metric value.
        should_stop: Flag indicating if training should stop.

    Example:
        >>> early_stopping = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     if early_stopping.step(val_loss):
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait.
            min_delta: Minimum improvement required.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Check if training should stop.

        Args:
            metric: Current metric value.

        Returns:
            True if training should stop.
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class ModelCheckpoint:
    """Callback for saving the best model during training.

    Saves the model whenever the monitored metric improves.

    Attributes:
        save_dir: Directory for saving checkpoints.
        monitor: Metric name to monitor.
        best_score: Best observed metric value.
        best_model_path: Path to the best saved model.

    Example:
        >>> checkpoint = ModelCheckpoint(save_dir="checkpoints")
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     checkpoint.step(model, val_loss)
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        """Initialize model checkpoint.

        Args:
            save_dir: Directory for saving models.
            monitor: Metric to monitor.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_model_path: Optional[str] = None

    def step(self, model: nn.Module, metric: float) -> bool:
        """Check and save if metric improved.

        Args:
            model: Model to potentially save.
            metric: Current metric value.

        Returns:
            True if model was saved.
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_model_path = str(self.save_dir / "best_model.pt")

            torch.save(model.state_dict(), self.best_model_path)
            logger.info(
                f"Saved best model with {self.monitor}={metric:.6f}"
            )
            return True

        return False

    def load_best(self, model: nn.Module) -> None:
        """Load the best saved model.

        Args:
            model: Model to load weights into.
        """
        if self.best_model_path is not None:
            model.load_state_dict(torch.load(self.best_model_path))
            logger.info(f"Loaded best model from {self.best_model_path}")
        else:
            logger.warning("No best model saved yet")

