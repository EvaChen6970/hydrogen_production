"""
Baseline models for green hydrogen production forecasting.
Implements:
  - LSTM  (methodology §II.D, ref [5])
  - GRU  (methodology §II.D, ref [5])
  - 1D-CNN with increasing dilation rates (methodology §II.D)
  - XGBoost with quantile regression (methodology §II.D, ref [6])

Each model returns (h2_output, lc_output) matching the Transformer interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


# ────────────────────────────────────────────────────────────────────────────────
# PyTorch-based baselines (LSTM, GRU, 1D-CNN)
# ────────────────────────────────────────────────────────────────────────────────

class _RNNBase(nn.Module):
    """Shared infrastructure for LSTM and GRU."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = None  # set in subclass
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Output heads: use the last hidden state
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.prod_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.lcoh_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, T, D)
        h = self.input_proj(x)                              # (batch, T, hidden)
        rnn_out, _ = self.rnn(h)                          # (batch, T, hidden*dirs)
        # Use last time step
        last = rnn_out[:, -1, :]                          # (batch, hidden*dirs)
        h2   = self.prod_head(last)
        lc   = self.lcoh_head(last)
        return h2, lc


class HydrogenLSTM(_RNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = nn.LSTM(
            input_size=kwargs.get("hidden_dim", 256),
            hidden_size=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 2),
            batch_first=True,
            dropout=kwargs.get("dropout", 0.1),
            bidirectional=kwargs.get("bidirectional", False),
        )


class HydrogenGRU(_RNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rnn = nn.GRU(
            input_size=kwargs.get("hidden_dim", 256),
            hidden_size=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 2),
            batch_first=True,
            dropout=kwargs.get("dropout", 0.1),
            bidirectional=kwargs.get("bidirectional", False),
        )


class _ConvBlock(nn.Module):
    """1-D conv block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv  = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn    = nn.BatchNorm1d(out_ch)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.drop(self.act(self.bn(self.conv(x))))
        return out + residual


class Hydrogen1DCNN(nn.Module):
    """
    1-D CNN with 4 convolutional blocks with increasing dilation rates,
    capturing multi-scale temporal patterns (methodology §II.D).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        dilations = [1, 2, 4, 8]   # multi-scale receptive field

        # Project input to hidden channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        in_ch = hidden_channels
        for i in range(num_blocks):
            self.blocks.append(_ConvBlock(in_ch, hidden_channels, kernel_size=5,
                                          dilation=dilations[i], dropout=dropout))
            in_ch = hidden_channels

        # Pool over time → (batch, hidden_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.prod_head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.lcoh_head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, T, D) → (batch, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).squeeze(-1)                # (batch, hidden)
        return self.prod_head(h), self.lcoh_head(h)


# ────────────────────────────────────────────────────────────────────────────────
# XGBoost baseline (handles numpy arrays, not torch tensors)
# ────────────────────────────────────────────────────────────────────────────────

class HydrogenXGBoost:
    """
    XGBoost wrapper with MultiOutputRegressor for concurrent
    hydrogen production and LCOH prediction.
    Uses quantile regression for uncertainty quantification (methodology §II.D).
    """

    def __init__(self, quantile: float = 0.5, **xgb_kwargs):
        self.quantile   = quantile
        self.xgb_kwargs = dict(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            device="cpu",
            **xgb_kwargs,
        )
        self.model = MultiOutputRegressor(
            xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                **self.xgb_kwargs,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HydrogenXGBoost":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ────────────────────────────────────────────────────────────────────────────────
# Registry for easy model construction
# ────────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "transformer": None,          # built separately in train.py
    "lstm":        HydrogenLSTM,
    "gru":         HydrogenGRU,
    "cnn":         Hydrogen1DCNN,
    "xgboost":     HydrogenXGBoost,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory function to build a model by name."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    if name == "transformer":
        from .transformer import HydrogenTransformer
        return HydrogenTransformer(**kwargs)
    return cls(**kwargs)


if __name__ == "__main__":
    for name, cls in [("lstm", HydrogenLSTM), ("gru", HydrogenGRU), ("cnn", Hydrogen1DCNN)]:
        model = cls(input_dim=10)
        x = torch.randn(4, 1440, 10)
        h2, lc = model(x)
        print(f"{name}: h2={h2.shape}, lc={lc.shape}, "
              f"params={sum(p.numel() for p in model.parameters()):,}")
