"""Deep learning models for energy consumption forecasting.

This module implements neural network architectures:
- LSTM: Long Short-Term Memory for sequential patterns
- CNN: Convolutional Neural Network for local patterns
- Transformer: Attention-based model for long-range dependencies

All models support optional temporal features and multi-step forecasting.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DeepModelConfig:
    """Configuration for deep learning models.

    Attributes:
        input_window: Number of time steps in input sequence.
        output_horizon: Number of steps to forecast.
        hidden_size: Hidden dimension size.
        num_layers: Number of recurrent/transformer layers.
        dropout: Dropout probability.
        n_temporal_features: Number of temporal features (0 if none).
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.
        num_epochs: Number of training epochs.
    """

    input_window: int = 120
    output_horizon: int = 7
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    n_temporal_features: int = 0
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting.

    Long Short-Term Memory networks are designed to capture
    long-range temporal dependencies through gating mechanisms
    that control information flow.

    Architecture:
        Input -> LSTM layers -> Fully Connected -> Output

    Attributes:
        config: Model configuration.
        lstm: LSTM layer stack.
        fc: Final fully connected layer.
        dropout: Dropout layer.

    Example:
        >>> config = DeepModelConfig(hidden_size=64, num_layers=2)
        >>> model = LSTMModel(config)
        >>> output = model(x_seq, temporal_features)
    """

    def __init__(self, config: DeepModelConfig) -> None:
        """Initialize the LSTM model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Input size: 1 (consumption) + temporal features
        input_size = 1 + config.n_temporal_features

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.output_horizon)

        logger.info(
            f"Created LSTM model: input_size={input_size}, "
            f"hidden_size={config.hidden_size}, "
            f"num_layers={config.num_layers}"
        )

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the LSTM.

        Args:
            x: Input sequence of shape (batch, seq_len) or (batch, seq_len, 1).
            features: Optional temporal features of shape (batch, seq_len, n_features).

        Returns:
            Predictions of shape (batch, output_horizon).
        """
        # Ensure x has feature dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Concatenate with temporal features if provided
        if features is not None:
            # Only use input window features
            input_features = features[:, : self.config.input_window, :]
            x = torch.cat([x, input_features], dim=-1)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)

        return out

    def get_intermediate_output(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the intermediate representation before final layer.

        Used for ensemble model to extract features.

        Args:
            x: Input sequence.
            features: Optional temporal features.

        Returns:
            Hidden representation of shape (batch, hidden_size).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if features is not None:
            input_features = features[:, : self.config.input_window, :]
            x = torch.cat([x, input_features], dim=-1)

        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]


class CNNModel(nn.Module):
    """1D CNN model for time series forecasting.

    Convolutional Neural Networks extract local patterns
    through sliding filters, effective for detecting
    short-term dependencies and motifs.

    Architecture:
        Input -> Conv1D blocks -> Global pooling -> FC -> Output

    Attributes:
        config: Model configuration.
        conv_layers: Convolutional layer stack.
        fc: Final fully connected layers.

    Example:
        >>> config = DeepModelConfig(hidden_size=64)
        >>> model = CNNModel(config)
        >>> output = model(x_seq, temporal_features)
    """

    def __init__(
        self,
        config: DeepModelConfig,
        kernel_sizes: List[int] = None,
        num_filters: List[int] = None,
    ) -> None:
        """Initialize the CNN model.

        Args:
            config: Model configuration.
            kernel_sizes: List of kernel sizes for each conv layer.
            num_filters: List of filter counts for each conv layer.
        """
        super().__init__()
        self.config = config

        kernel_sizes = kernel_sizes or [7, 5, 3]
        num_filters = num_filters or [32, 64, 128]

        input_size = 1 + config.n_temporal_features

        # Build convolutional layers
        layers = []
        in_channels = input_size

        for i, (k, f) in enumerate(zip(kernel_sizes, num_filters)):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=f,
                    kernel_size=k,
                    padding=k // 2,
                )
            )
            layers.append(nn.BatchNorm1d(f))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_channels = f

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling will reduce to (batch, num_filters[-1])
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.output_horizon),
        )

        logger.info(
            f"Created CNN model: filters={num_filters}, "
            f"kernels={kernel_sizes}"
        )

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input sequence of shape (batch, seq_len).
            features: Optional temporal features.

        Returns:
            Predictions of shape (batch, output_horizon).
        """
        # Ensure x has feature dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Concatenate with temporal features if provided
        if features is not None:
            input_features = features[:, : self.config.input_window, :]
            x = torch.cat([x, input_features], dim=-1)

        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv_layers(x)

        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Fully connected
        out = self.fc(x)

        return out

    def get_intermediate_output(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the intermediate representation before final layer.

        Args:
            x: Input sequence.
            features: Optional temporal features.

        Returns:
            Hidden representation.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if features is not None:
            input_features = features[:, : self.config.input_window, :]
            x = torch.cat([x, input_features], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models.

    Adds positional information to input embeddings using
    sinusoidal functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting.

    Transformers use self-attention to capture long-range
    dependencies without the sequential bottleneck of RNNs.

    Architecture:
        Input -> Embedding -> Positional Encoding -> 
        Transformer Encoder -> FC -> Output

    Attributes:
        config: Model configuration.
        embedding: Input embedding layer.
        pos_encoder: Positional encoding.
        transformer: Transformer encoder layers.
        fc: Final fully connected layer.

    Example:
        >>> config = DeepModelConfig(hidden_size=64, num_layers=2)
        >>> model = TransformerModel(config)
        >>> output = model(x_seq, temporal_features)
    """

    def __init__(
        self,
        config: DeepModelConfig,
        nhead: int = 4,
    ) -> None:
        """Initialize the Transformer model.

        Args:
            config: Model configuration.
            nhead: Number of attention heads.
        """
        super().__init__()
        self.config = config

        input_size = 1 + config.n_temporal_features
        d_model = config.hidden_size

        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * (d_model // nhead + 1)

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=config.input_window + config.output_horizon)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.fc = nn.Linear(d_model, config.output_horizon)
        self.d_model = d_model

        logger.info(
            f"Created Transformer model: d_model={d_model}, "
            f"nhead={nhead}, num_layers={config.num_layers}"
        )

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            x: Input sequence of shape (batch, seq_len).
            features: Optional temporal features.

        Returns:
            Predictions of shape (batch, output_horizon).
        """
        # Ensure x has feature dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Concatenate with temporal features if provided
        if features is not None:
            input_features = features[:, : self.config.input_window, :]
            x = torch.cat([x, input_features], dim=-1)

        # Embed and add positional encoding
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer(x)

        # Use mean pooling across sequence
        x = x.mean(dim=1)

        # Final prediction
        out = self.fc(x)

        return out

    def get_intermediate_output(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the intermediate representation before final layer.

        Args:
            x: Input sequence.
            features: Optional temporal features.

        Returns:
            Hidden representation.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if features is not None:
            input_features = features[:, : self.config.input_window, :]
            x = torch.cat([x, input_features], dim=-1)

        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        return x.mean(dim=1)


def create_model(
    model_type: str,
    config: DeepModelConfig,
) -> nn.Module:
    """Factory function to create deep learning models.

    Args:
        model_type: One of 'lstm', 'cnn', 'transformer'.
        config: Model configuration.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    model_type = model_type.lower()

    if model_type == "lstm":
        return LSTMModel(config)
    elif model_type == "cnn":
        return CNNModel(config)
    elif model_type == "transformer":
        return TransformerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

