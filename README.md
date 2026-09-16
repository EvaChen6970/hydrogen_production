# Green Hydrogen Production — Deep Learning Project

> Based on the NLR (National Laboratory of the Rockies) Historical Wind Electrolysis Dataset,
> this project implements a Transformer-based multi-task learning framework that jointly
> predicts PEM electrolyzer hydrogen production and the Levelized Cost of Hydrogen (LCOH).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Model Inputs and Outputs](#3-model-inputs-and-outputs)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [How to Run](#7-how-to-run)
8. [File Structure](#8-file-structure)
9. [Known Issues and Future Work](#9-known-issues-and-future-work)

---

## 1. Project Overview

**Goal**: leverage telemetry data from a wind-electrolyzer coupled system to train
deep learning models that simultaneously predict two key quantities:

| Target | Description | Physical Meaning |
|--------|-------------|------------------|
| Hydrogen production `H₂` | kg/h | Actual mass-flow output rate of the electrolyzer |
| Levelized Cost of Hydrogen `LCOH` | $/kg | Full life-cycle unit cost (CAPEX + electricity + water + degradation) |

**Core Challenges**:
- H₂ production spans 0.02–21.3 kg/h with high volatility (MAPE is sensitive to low values)
- LCOH spans 71–525 $/kg and is highly non-linear with respect to production rate
- Raw data contains only 28,800 samples (8 h × 2 experiments @ 1 Hz) — a small-sample regime

---

## 2. Dataset

### 2.1 Raw Data Source

| Field | Value |
|-------|-------|
| Dataset | NLR Simulated Wind Electrolysis Dataset |
| Link | `https://data.nlr.gov/submissions/305` |
| Equipment | Nel Hydrogen MC250 PEM electrolyzer (1.25 MW), GE 1.5 MW wind turbine |
| Sampling Rate | 1 Hz (raw), kept at 1 Hz after preprocessing |
| Experiments | 12 simulated wind profiles plus 1 characterization profile |
| Total Samples | 61,285 samples @ 1 Hz |

### 2.2 Column Mapping (CSV → Methodology Format)

> ⚠️ **Important**: The CSV column `H2E_f_NLR_CurrentCmd (%)` has `%` in its name but stores **amperes** (e.g. 2097, 2539 A), not percentages. The code internally maps it to `dc_stack_current`. The `★` marker below highlights this discrepancy.

| Methodology Variable | Symbol | Actual CSV Column | Notes |
|----------------------|--------|-------------------|-------|
| Wind speed | v (m/s) | *(derived, no original column)* | Inverse GE 1.5 MW power curve from `GE 1.5 MW Wind Turbine Power (kW)` |
| Turbulence intensity | TI (%) | *(derived, no original column)* | 60-s rolling std of wind speed divided by rolling mean × 100 |
| DC stack current | I_DC (A) | `H2E_f_NLR_CurrentCmd (%)` ★ | Column named with `%` but stores amperes; provided by the actual `H2E_f_NLR_CurrentCmd (A)` column (CSV column #33) |
| Current ramp rate | Ċ (A/s) | *(derived, no original column)* | Mapped from `Experiment`: `wind-GE1.5MW_0.65-200.csv` → 200, `wind-GE1.5MW_0.65-400.csv` → 400 |
| Stack temperature | T_s (°C) | `H2E_f_TE218_Temp` | Direct CSV column ✓ |
| Stack pressure | P_s (bar) | `H2E_f_PT264_Pressure` | Direct CSV column ✓ |
| Power consumption | P_el (kW) | `H2E_n_PSU_A_Power (kW)` | Direct CSV column ✓ |

**Derived Columns Summary** (computed in `data_loader.py`; no corresponding raw CSV column):

| Derived Column | Computation |
|----------------|-------------|
| `Wind_Speed_ms` | Inverse GE 1.5 MW power curve: `p = power/1500`, `v = p × 7.5 + 3.5`, clipped to [3.5, 12] m/s |
| `Turbulence_pct` | `rolling_std(wind_speed, 60) / rolling_mean(wind_speed, 60) × 100`, clipped to [0, 50]% |
| `Current_Ramp_As` | Mapped from experiment file name (200 or 400 A/s) |

### 2.3 LCOH Computation

Following the EU Hydrogen Observatory LCOH calculator methodology:

```
LCOH ≈ CAPEX_ann + electricity_cost + water_cost + degradation_cost

where:
  electricity_cost = efficiency_kWh/kg × electricity_price_$/kWh
  water_cost      = 9 L H₂O / kg H₂ × water_price_$/L
  CAPEX_ann       = CAPEX ($/kg) / project_life (yr)
```

**Default Parameters**: CAPEX = 700 $/kg, lifetime = 10 yr, electricity price = 0.06 $/kWh, water price = 0.004 $/m³.

---

## 3. Model Inputs and Outputs

### 3.1 Input Format

```
Input Tensor:  (batch_size, T, D) = (batch_size, 360, 10)
  T = 360   # look-back window: 360 seconds (6 minutes of history)
  D = 10     # input channel count (see table below)

Channel Order (fixed):
  0: wind_speed        # Wind speed (m/s); derived feature, no raw CSV column
  1: turbulence        # Turbulence intensity (%); derived feature, no raw CSV column
  2: dc_stack_current  # DC stack current (A); from CSV column `H2E_f_NLR_CurrentCmd (%)` (column named with `%` but data is amperes)
  3: current_ramp_rate # Current ramp rate (A/s); derived from experiment file name
  4: stack_temperature # Stack temperature (°C); CSV column `H2E_f_TE218_Temp` ✓
  5: stack_pressure    # Stack pressure (bar); CSV column `H2E_f_PT264_Pressure` ✓
  6: power_consumption # Electrolysis power consumption (kW)
  7: efficiency_kwh_kg # Electrolysis efficiency (kWh/kg); CSV column `Efficiency (kWh/kg)` ✓
  8: h2_history        # Historical H2 flow (kg/h); past values only
  9: experiment_progress # Normalized position within each experiment
```

**Preprocessing**:
- Missing values: forward-fill (≤5 consecutive samples), linear interpolation (longer gaps)
- Normalization: `StandardScaler` fit on the training set only; validate/test use only `transform` (no data leakage)
- Normalized range: all channels mapped to mean ≈ 0, std ≈ 1

### 3.2 Output Format

```
Output Tensor:  (batch_size, 2)
  0: h2_production (kg/h)   # H₂ output forecast at the end of the next 60-second horizon
  1: lcost ($/kg)           # corresponding LCOH estimate
```

**Target Ground-Truth Sources**:

| Target | Symbol | Source | CSV Column |
|--------|--------|--------|------------|
| Hydrogen production | H₂ (kg/h) | **Direct CSV measurement** | `IVAL_f_FM011_Flow (kg/hr)` (Emerson Coriolis flow meter) ✓ |
| Levelized Cost of Hydrogen | LCOH ($/kg) | **Computed in code** (no original CSV column) | Calculated row-by-row from the `Efficiency (kWh/kg)` column using the EU Hydrogen Observatory formula |

### 3.3 Temporal Window Strategy

```
Raw sequence length N = 28,800

Sliding-window parameters:
  lookback T  = 360   → use the past 360 s as input
  horizon  H  = 60    → forecast the next 60 s
  stride       = 30    → generate a new window every 30 s

Dataset split (temporal, no shuffling):
  Train: 70% → 659 windows
  Val:   15% → 131 windows
  Test:  15% → 131 windows
```

---

## 4. Model Architecture

### 4.1 Main Model: Transformer (methodology §II.B)

```
Input (batch, 360, 10)
    │
    ▼
InputEncoder  (Linear: 10→128 + BatchNorm1d)
    │ (batch, 360, 128)
    ▼
PositionalEncoding  (learnable, initialized from N(0, 0.02))
    │ (batch, 360, 128)
    ▼
Transformer Encoder × 3 layers
    ├─ Multi-Head Self-Attention (4 heads, pre-norm, GELU)
    ├─ Feed-Forward (128→512→128, GELU, dropout=0.1)
    └─ Residual connections
    │ (batch, 360, 128)
    ▼
Global average pooling → (batch, 128)
    │
    ├─────────────────────────────────┐
    ▼                                 ▼
Production Head                   LCOH Head
  Linear(128→128, GELU)            Linear(128→128, GELU)
  Dropout(0.1)                     Dropout(0.1)
  Linear(128→1)                    + skip connection (from Production hidden layer)
                                    Linear(128→1)
    │                                 │
    ▼                                 ▼
h2_output (batch, 1)          lc_output (batch, 1)
```

**Default Hyperparameters**:

| Parameter | Default | Search Range |
|-----------|---------|--------------|
| `d_model` | 128 | {64, 128, 256} |
| `num_heads` | 4 | {4, 8} |
| `num_layers` | 3 | {2, 3, 4} |
| `dim_feedforward` | 256 | {128, 256, 512} |
| `dropout` | 0.1 | — |
| Total parameters | ~807,042 | — |

### 4.2 Baseline Models (methodology §II.D)

| Model | Structure | Parameters |
|-------|-----------|------------|
| **LSTM** | 2-layer LSTM, 256 hidden units, dual-task MLP heads | ~1,120,770 |
| **GRU** | 2-layer GRU, 256 hidden units, dual-task MLP heads | ~857,602 |
| **1D-CNN** | 4-layer dilated convolutions (dilation: 1, 2, 4, 8), multi-scale receptive field | ~369,154 |
| **XGBoost** | 300 trees, max_depth=8, quantile regression | — |

### 4.3 Multi-Task Loss (Kendall et al. 2018)

```
L_total = (1 / σ²_H2) · MSE_H2 + (1 / σ²_LCOH) · MSE_LCOH
          + log(σ_H2) + log(σ_LCOH)

where σ²_H2 and σ²_LCOH are learnable parameters (auto-balancing the two tasks)
```

**Initialization**: `log_var_H2 = 4.0` (large value → small precision → stable startup, prevents NaN).

---

## 5. Training Pipeline

### 5.1 Optimizer Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay 0.01 mitigates overfitting |
| Initial learning rate | 1×10⁻⁴ | — |
| Warm-up | 10 epochs | Linear ramp-up to peak LR |
| LR schedule | Cosine Annealing | Cosine decay to 1×10⁻⁶ after warm-up |
| Gradient clipping | max_norm = 1.0 | Stabilizes training |
| Batch size | 32 | Safe value for CPU environments |
| Max epochs | 200 | — |
| Early-stopping patience | 30 epochs | Stops when val MAE stops improving |
| Mixed precision | Disabled (CPU) | Enable on CUDA-enabled devices |

### 5.2 Data Preprocessing Steps

```
Raw CSV (1 Hz, 28,800 rows)
  ├─ Add experiment metadata (current ramp rate)
  ├─ Derive wind speed (simplified GE 1.5 MW power curve)
  ├─ Compute turbulence intensity (60-s rolling std)
  ├─ Handle missing values (ffill ≤5 + linear interpolation)
  ├─ Compute LCOH (row-by-row with economic parameters)
  └─ Sliding-window → (T=360, H=60, stride=30)
```

---

## 6. Evaluation Metrics

Each target (H₂ production, LCOH) reports the following metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| MAE | mean\|ŷ − y\| | Mean Absolute Error |
| RMSE | √mean((ŷ − y)²) | Root Mean Squared Error; more sensitive to large errors |
| MAPE | mean\|ŷ − y\| / \|y\| × 100% | Mean Absolute Percentage Error; volatile when targets are near zero |
| R² | 1 − SS_res / SS_tot | Coefficient of determination; <0 means worse than predicting the mean |

---

## 7. How to Run

### 7.1 Dependencies

```bash
pip install torch numpy pandas scikit-learn xgboost
```

> Note: PyTorch 2.8.0+ is already installed; no extra installation required.

### 7.2 Download the Dataset (already completed)

```bash
# Data file already downloaded to:
# E:\pythonProject\yfel\data\raw\combined_historical_wind_experiments.csv
```

To re-download manually:
```python
# URL: https://data.nlr.gov/system/files/316/1774293076-combined_historical_wind_experiments.csv
```

### 7.3 Run the Full Pipeline

```bash
# Train all models
python main.py

# Train only the Transformer (faster)
python main.py --model transformer --epochs 50

# Train a single baseline
python main.py --model lstm    --epochs 50
python main.py --model gru     --epochs 50
python main.py --model cnn     --epochs 50
python main.py --model xgboost --epochs 50

# Evaluate the published NLR PT-MELT LSTM baseline
python scripts/evaluate_ptmelt.py --horizon 1
python scripts/evaluate_ptmelt.py --horizon 30

# Compare PT-MELT with the project Transformer and create a plot
python scripts/compare_predictions.py
```

The PT-MELT model is a separately published one-step hydrogen-flow model. It
uses the 15 raw electrolysis sensors listed in `data/ptmelt_README.md`, so its
metrics should be compared with the one-step command first. The 30-second run
is an additional stress test, not the model's original training objective.

### 7.4 Modify Hyperparameters

Edit the `CONFIG` dictionary in `main.py`. Example:

```python
CONFIG = dict(
    lookback    = 180,     # 3-min look-back (faster but less context)
    horizon     = 30,      # 30-s forecast
    d_model     = 64,      # smaller model
    num_layers  = 2,
    max_epochs  = 100,
    ...
)
```

---

## 8. File Structure

```
e:\pythonProject\yfel\
├── methodology.docx                      # Methodology paper (original project brief)
├── main.py                               # Entry script (training + evaluation + results summary)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Data loading, preprocessing, Dataset, DataLoader
│   ├── trainer.py                        # Trainer, early stopping, LR schedule, evaluation
│   └── models/
│       ├── __init__.py
│       ├── transformer.py                # HydrogenTransformer + MultiTaskLoss
│       └── baselines.py                  # LSTM, GRU, 1D-CNN, XGBoost
│
├── data/
│   └── raw/
│       └── combined_historical_wind_experiments.csv   # Raw data (downloaded)
│
├── outputs/                              # Training outputs (auto-created)
│   ├── transformer_test_predictions.csv  # Test targets and predictions
│   └── figures/transformer_test_predictions.png # H2/LCOH comparison plot
│   ├── transformer_best.pt               # Best Transformer weights
│   ├── lstm_best.pt                      # Best LSTM weights
│   ├── ...                               # Other model weights
│   └── results.json                      # Final evaluation metrics for every model
│
└── README.md                             # This file
```

---

## 9. Known Issues and Future Work

### 9.1 Current Limitations

| Issue | Cause | Improvement Direction |
|-------|-------|-----------------------|
| High MAPE (174%) | Relative error explodes at very low H₂ outputs (<1 kg/h); the dataset only spans 8 hours so the sample distribution is highly imbalanced | Add more data; consider quantile regression or piecewise modelling |
| R² < 0 | Too few training windows (659); the model cannot learn complex non-linearities | Apply data augmentation (e.g. noise injection); reduce model complexity |
| Poor LCOH prediction | LCOH changes slowly (driven mainly by steady-state efficiency); short-term volatility is hard to capture | Model LCOH separately, or add external electricity-price/CAPEX features |

### 9.2 Recommended Improvements

1. **Get more data**: NLR also publishes Solar PV electrolysis data (`data.nlr.gov/submissions/318`), which can be merged for more samples.
2. **Down-sampling strategy**: Resample from 1 Hz to 0.1 Hz (1 point every 10 s) and grow the look-back window to T = 1440 (4 hours).
3. **HuggingFace models**: NLR provides pretrained models on HuggingFace (`NatLabRockies/ptmelt-hydrogen-electrolysis`) that can be used as initialization weights or as a strong baseline.
4. **Hyperparameter search**: Run Bayesian optimization with `optuna` over `d_model`, `num_layers`, `learning_rate`, etc.
5. **Interpretability**: Visualize attention-weight heatmaps to understand which timesteps and features the model focuses on.

---

## References

[1] NLR, "Public Reference Data for Megawatt-Scale Hydrogen Electrolysis – Simulated Wind," 2026.  
[2] Clean Hydrogen JU, "Manual – Levelised Cost of Hydrogen (LCOH) Calculator," European Hydrogen Observatory, 2024.  
[3] Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.  
[4] Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses," CVPR, 2018.  
[5] Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.  
[6] Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.  
[7] Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting," AAAI, 2021.  
[8] Ullah et al., "Sequential Gated Recurrent and Self-Attention Explainable Deep Learning Model for Predicting Hydrogen Production," Applied Energy, 2025.
