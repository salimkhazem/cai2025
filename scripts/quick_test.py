#!/usr/bin/env python
"""Quick test script to verify the installation works.

Usage:
    python scripts/quick_test.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    from src.data import UCLDataLoader, ODREDataLoader, DataConfig
    print("  ✓ Data modules")
    
    from src.features import TemporalFeatureEngine, CyclicalEncodingConfig
    print("  ✓ Feature modules")
    
    from src.models import LSTMModel, CNNModel, TransformerModel, MetaClassifier
    print("  ✓ Model modules")
    
    from src.training import Trainer, TrainerConfig
    print("  ✓ Training modules")
    
    from src.evaluation import compute_all_metrics, MetricsResult
    print("  ✓ Evaluation modules")
    
    from src.visualization import plot_per_day_metrics
    print("  ✓ Visualization modules")
    
    print("\nAll imports successful!")


def test_cyclical_encoding():
    """Test cyclical encoding."""
    import numpy as np
    from src.features import CyclicalEncoder
    
    print("\nTesting cyclical encoding...")
    
    encoder = CyclicalEncoder()
    
    # Test day of week encoding
    days = np.array([0, 1, 2, 3, 4, 5, 6])
    sin_enc, cos_enc = encoder.encode(days, max_val=7)
    
    # Sunday (0) should be close to Saturday (6)
    dist_sun_sat = np.sqrt((sin_enc[0] - sin_enc[6])**2 + (cos_enc[0] - cos_enc[6])**2)
    dist_sun_mon = np.sqrt((sin_enc[0] - sin_enc[1])**2 + (cos_enc[0] - cos_enc[1])**2)
    
    assert dist_sun_sat < 1.0, "Sunday should be close to Saturday"
    print(f"  ✓ Distance Sunday-Saturday: {dist_sun_sat:.4f}")
    print(f"  ✓ Distance Sunday-Monday: {dist_sun_mon:.4f}")
    
    print("\nCyclical encoding works correctly!")


def test_model_creation():
    """Test model instantiation."""
    import torch
    from src.models.deep_models import DeepModelConfig, LSTMModel, CNNModel
    from src.models.ensemble import EnsembleConfig, MetaClassifier
    
    print("\nTesting model creation...")
    
    config = DeepModelConfig(
        input_window=120,
        output_horizon=7,
        hidden_size=32,
        num_layers=1,
        n_temporal_features=10,
    )
    
    # Test LSTM
    lstm = LSTMModel(config)
    x = torch.randn(4, 120)
    features = torch.randn(4, 127, 10)
    out = lstm(x, features)
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"
    print(f"  ✓ LSTM output shape: {out.shape}")
    
    # Test CNN
    cnn = CNNModel(config)
    out = cnn(x, features)
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"
    print(f"  ✓ CNN output shape: {out.shape}")
    
    # Test Meta-Classifier
    ensemble_config = EnsembleConfig(
        input_window=120,
        output_horizon=7,
        lstm_hidden=32,
        lstm_layers=1,
        n_temporal_features=10,
    )
    mc = MetaClassifier(ensemble_config)
    out = mc(x, features)
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"
    print(f"  ✓ Meta-Classifier output shape: {out.shape}")
    
    print("\nAll models work correctly!")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Cyclical Energy Forecasting - Quick Test")
    print("=" * 50)
    
    test_imports()
    test_cyclical_encoding()
    test_model_creation()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()

