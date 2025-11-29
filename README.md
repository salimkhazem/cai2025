# Cyclical Temporal Encoding for Ensemble Deep Learning in Multistep Energy Forecasting

<p align="center">
  <a href="https://www.ieeesmc.org/cai-2026/"><img src="https://img.shields.io/badge/IEEE-CAI%202026-blue.svg" alt="Conference"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776ab.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#installation">Installation</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="results/figures/architecture.png" alt="Architecture" width="800">
</p>

---

## Abstract

> Accurate electricity consumption forecasting is essential for smart grid operations and energy planning. This work presents a deep learning-based framework that combines temporal feature engineering with architectural complementarity for improved short-term predictions. We extract and transform calendar-based features using **sine and cosine encodings** to capture periodic patterns. To address both long-term and local temporal dependencies, we employ an ensemble of two base models: a **Long Short-Term Memory (LSTM)** network and a **Convolutional Neural Network (CNN)**. Their outputs are integrated through a **Meta-Classifier** composed of multiple MLP regressors, each dedicated to one forecast day. Experiments on multiple real-world datasets demonstrate that our hybrid model consistently outperforms individual baselines across a 7-day prediction horizon, achieving an average MAE improvement of **13.2%** over the best individual model.

## Highlights

- **Cyclical Temporal Encoding**: Sine/cosine transformations preserve temporal continuity
- **Meta-Classifier Ensemble**: Day-specific MLPs combine LSTM and CNN features optimally
- **Comprehensive Evaluation**: 6+ baselines on 2 real-world datasets
- **Ablation Studies**: Systematic analysis of each component's contribution
- **Full Reproducibility**: Complete codebase with configs and scripts

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA (optional, for GPU acceleration)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/talan-research/cyclical-energy-forecasting.git
cd cyclical-energy-forecasting

# Create environment
uv venv 
# activate the env 
source .venv/bin/activate 
# install requirements using uv
uv sync

# using requirements.txt
uv pip install -r requirements.txt
```

### Verify Installation

```bash
python scripts/quick_test.py
```

---

## Datasets

We evaluate on three publicly available datasets:

| Dataset | Source | Period | Granularity | Samples |
|---------|--------|--------|-------------|---------|
| **UCI Electricity** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) | 2011-2014 | 15 min → Daily | 1,462 days |
| **ODRE French** | [Open Data Réseaux Énergies](https://opendata.reseaux-energies.fr/) | 2019-2023 | 30 min → Daily | ~1,500 days |
| **French Enedis** | | | | | 

### Download & Prepare Data

```bash
# UCI Electricity dataset
mkdir -p input/UCL_dataset
wget -O input/UCL_dataset/LD2011_2014.txt \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
unzip input/UCL_dataset/LD2011_2014.txt.zip -d input/UCL_dataset/

# ODRE dataset (optional)
# Download from: https://opendata.reseaux-energies.fr/
```

---

## Usage

### Training & Evaluation

```bash
# Run all experiments (baselines + deep learning + ablation)
python scripts/run_experiments.py --experiment all --dataset ucl --seed 42

# Run specific experiments
python scripts/run_experiments.py --experiment baseline   # Traditional methods
python scripts/run_experiments.py --experiment deep       # Deep learning models
python scripts/run_experiments.py --experiment ablation   # Ablation studies
```

### Configuration

Experiments are configured via YAML files in `configs/`:

```yaml
# configs/default.yaml
data:
  input_window: 120    # Days of history
  output_horizon: 7    # Days to forecast
  
models:
  lstm:
    hidden_size: 64
    num_layers: 2
    
training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 100
```

### Custom Experiments

```python
from src.data import UCLDataLoader, DataConfig
from src.features import TemporalFeatureEngine
from src.models import MetaClassifier
from src.training import Trainer

# Load data
loader = UCLDataLoader(config=DataConfig(input_window=120, output_horizon=7))
loader.load("input/UCL_dataset/LD2011_2014.txt")

# Extract features
engine = TemporalFeatureEngine()
features = engine.fit_transform(loader.data.index, loader.data["consumption"])

# Train model
model = MetaClassifier(config)
trainer = Trainer(model)
trainer.fit(train_loader, val_loader)
```

---

## Repository Structure

```
.
├── configs/                    # Experiment configurations
│   └── default.yaml
├── input/                      # Dataset directory
│   ├── UCL_dataset/
│   └── odre_data/
├── notebooks/                  # Analysis notebooks
│   └── plot_results.ipynb
├── paper/                      # LaTeX source
│   └── main.tex
├── results/                    # Experiment outputs
│   ├── metrics/               # CSV results
│   ├── figures/               # Generated plots
│   └── correlations/
├── scripts/                    # Executable scripts
│   ├── run_experiments.py
│   └── quick_test.py
├── src/                        # Source code
│   ├── data/                  # Data loading
│   ├── features/              # Temporal encoding
│   ├── models/                # Model architectures
│   ├── training/              # Training pipeline
│   ├── evaluation/            # Metrics
│   └── visualization/         # Plotting
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---