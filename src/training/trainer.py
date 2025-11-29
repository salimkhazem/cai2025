"""Training pipeline for deep learning models.

This module provides a comprehensive training framework with:
- Flexible model training loop
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Logging and metrics tracking
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the training pipeline.

    Attributes:
        learning_rate: Initial learning rate.
        batch_size: Training batch size.
        num_epochs: Maximum number of training epochs.
        optimizer: Optimizer type ('adam', 'adamw', 'sgd').
        weight_decay: L2 regularization strength.
        scheduler: Learning rate scheduler ('plateau', 'cosine', None).
        patience: Early stopping patience.
        min_delta: Minimum improvement for early stopping.
        device: Training device ('cuda', 'cpu', 'auto').
        checkpoint_dir: Directory for saving checkpoints.
        log_interval: Steps between logging.
        gradient_clip: Maximum gradient norm for clipping.
    """

    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    scheduler: Optional[str] = "plateau"
    patience: int = 10
    min_delta: float = 1e-4
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    gradient_clip: Optional[float] = 1.0


class Trainer:
    """Training pipeline for PyTorch models.

    This class handles the complete training workflow including:
    - Data loading and batching
    - Forward/backward passes
    - Optimizer steps
    - Validation evaluation
    - Early stopping and checkpointing

    Attributes:
        model: PyTorch model to train.
        config: TrainerConfig with hyperparameters.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        device: Training device.
        history: Training history dictionary.

    Example:
        >>> trainer = Trainer(model, config)
        >>> history = trainer.fit(train_loader, val_loader)
        >>> trainer.save_model("best_model.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            config: Training configuration.
            criterion: Loss function (default: MSELoss).
        """
        self.config = config or TrainerConfig()
        self.model = model

        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Set loss function
        self.criterion = criterion or nn.MSELoss()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint = ModelCheckpoint(
            save_dir=self.config.checkpoint_dir,
            monitor="val_loss",
        )

        logger.info(f"Trainer initialized on device: {self.device}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer based on config.

        Returns:
            Optimizer instance.
        """
        params = self.model.parameters()

        if self.config.optimizer.lower() == "adam":
            return Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adamw":
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler based on config.

        Returns:
            Scheduler instance or None.
        """
        if self.config.scheduler is None:
            return None
        elif self.config.scheduler.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )
        elif self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch - handle both (x, y) and (x, features, y) formats
            if len(batch) == 3:
                x, features, y = batch
                features = features.to(self.device)
            else:
                x, y = batch
                features = None

            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x, features)
            loss = self.criterion(predictions, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.debug(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        return total_loss / n_batches

    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch - handle both (x, y) and (x, features, y) formats
                if len(batch) == 3:
                    x, features, y = batch
                    features = features.to(self.device)
                else:
                    x, y = batch
                    features = None

                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x, features)
                loss = self.criterion(predictions, y)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            Training history dictionary.
        """
        logger.info(
            f"Starting training for {self.config.num_epochs} epochs "
            f"on {self.device}"
        )
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rate"].append(current_lr)

                # Checkpointing
                self.checkpoint.step(self.model, val_loss)

                # Early stopping
                if self.early_stopping.step(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s"
                )
            else:
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, Time: {epoch_time:.1f}s"
                )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        # Load best model
        if val_loader is not None and self.checkpoint.best_model_path is not None:
            self.load_model(self.checkpoint.best_model_path)
            logger.info(f"Loaded best model from {self.checkpoint.best_model_path}")

        return self.history

    def predict(
        self,
        data_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions on a dataset.

        Args:
            data_loader: Data loader for prediction.

        Returns:
            Tuple of (predictions, targets) arrays.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch - handle both (x, y) and (x, features, y) formats
                if len(batch) == 3:
                    x, features, y = batch
                    features = features.to(self.device)
                else:
                    x, y = batch
                    features = None

                x = x.to(self.device)

                predictions = self.model(x, features)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y.numpy())

        return np.vstack(all_preds), np.vstack(all_targets)

    def save_model(self, path: str) -> None:
        """Save the model state.

        Args:
            path: Path to save the model.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load the model state.

        Args:
            path: Path to the saved model.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # Handle both formats: dict with 'model_state_dict' key or direct state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "history" in checkpoint:
                self.history = checkpoint["history"]
        else:
            # Direct state_dict from ModelCheckpoint callback
            self.model.load_state_dict(checkpoint)
        logger.info(f"Model loaded from {path}")

