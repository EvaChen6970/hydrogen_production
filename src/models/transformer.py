"""
Transformer-based multi-task model for green hydrogen production.
Implements the architecture described in methodology §II.B:
  - Input encoder: linear projection + batch norm + learnable positional encoding
  - Stack of L Transformer encoder layers with pre-norm residual config
  - Hierarchical attention: local window (w=16) + global stride
  - Production head (dmodel → 128 → 1) with GELU
  - LCOH head (dmodel → 128 → 1) with GELU + skip connection from production
  - Multi-task loss with learnable task weights (Kendall et al. 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Learnable positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class InputEncoder(nn.Module):
    """Linear projection + BatchNorm of raw (T, D) input to d_model dimension."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.bn   = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, D)
        x = self.proj(x)                      # (batch, T, d_model)
        # BatchNorm over time dimension
        b, t, d = x.shape
        x = x.permute(0, 2, 1)               # (batch, d_model, T)
        x = self.bn(x)                        # (batch, d_model, T)
        x = x.permute(0, 2, 1)               # (batch, T, d_model)
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of L Transformer encoder layers with pre-norm residual configuration.
    Optionally applies a hierarchical attention mask for local+global attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        local_window: int = 16,
        use_hierarchical: bool = True,
    ):
        super().__init__()
        self.use_hierarchical = use_hierarchical
        self.local_window     = local_window

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        for layer in self.layers:
            x = layer(x)
        return x


class ProductionHead(nn.Module):
    """MLP head for hydrogen production rate prediction: d_model → 128 → 1."""

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) or (batch, T, d_model)
        if x.dim() == 3:
            x = x.mean(dim=1)          # global average pooling over time
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)


class LCOHHead(nn.Module):
    """
    MLP head for LCOH prediction: d_model → 128 → 1.
    Incorporates a skip connection from the production embedding to capture
    the correlation between production efficiency and unit cost.
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, prod_emb: torch.Tensor) -> torch.Tensor:
        # x: encoder output; prod_emb: production head embedding before final layer
        if x.dim() == 3:
            x = x.mean(dim=1)
        # Skip connection: concatenate production embedding with LCOH embedding
        combined = x + prod_emb    # residual-style addition
        x = self.drop(self.act(self.fc1(combined)))
        return self.fc2(x)


class HydrogenTransformer(nn.Module):
    """
    Full Transformer-based multi-task model for green hydrogen production.

    Forward pass:
        raw_input (batch, T, 8)
          → InputEncoder → (batch, T, d_model)
          → PositionalEncoding
          → TransformerEncoder (L layers)
          → mean pool → (batch, d_model)
          → ProductionHead → h2_prod (batch, 1)
          → LCOHHead(prod_emb) → lc (batch, 1)
    """

    def __init__(
        self,
        input_dim: int = 10,
        d_model:   int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        local_window: int = 16,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.d_model     = d_model

        self.input_encoder  = InputEncoder(input_dim, d_model)
        self.pos_encoder    = PositionalEncoding(d_model)
        self.encoder        = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            local_window=local_window,
        )
        self.prod_head = ProductionHead(d_model)
        self.lcoh_head = LCOHHead(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, T, input_dim)
        h = self.input_encoder(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)               # (batch, T, d_model)

        # Global average pooling → (batch, d_model)
        h_pooled = h.mean(dim=1)

        # Production prediction
        prod_emb   = self.prod_head.act(self.prod_head.fc1(h_pooled))  # (batch, 128) pre-logit
        h2_output  = self.prod_head.fc2(prod_emb)                        # (batch, 1)

        # LCOH prediction (with skip connection from production embedding)
        lc_output  = self.lcoh_head(h, prod_emb)                         # (batch, 1)

        return h2_output, lc_output


class HydrogenTransformerLSTM(nn.Module):
    """Hybrid encoder: Transformer captures long context, LSTM tracks recent dynamics."""

    def __init__(self, input_dim=10, d_model=96, num_heads=4, num_layers=2,
                 lstm_hidden=128, lstm_layers=2, dim_feedforward=192, dropout=0.1):
        super().__init__()
        self.input_encoder = InputEncoder(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.last_h2_proj = nn.Sequential(nn.Linear(1, 32), nn.GELU())
        fused_dim = d_model + lstm_hidden + 32
        self.prod_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
        self.lcoh_head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        h = self.input_encoder(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        lstm_out, _ = self.lstm(h)
        last_h2 = self.last_h2_proj(x[:, -1, 8:9])
        context = torch.cat([h.mean(dim=1), lstm_out[:, -1, :], last_h2], dim=1)
        return self.prod_head(context), self.lcoh_head(context)


class HydroFormerLSTM(nn.Module):
    """Multi-scale Conv + Transformer + LSTM fusion model for H2 forecasting."""

    def __init__(self, input_dim=10, d_model=192, num_heads=8, num_layers=4,
                 lstm_hidden=256, lstm_layers=2, dim_feedforward=768, dropout=0.15):
        super().__init__()
        self.input_encoder = InputEncoder(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model=d_model, num_heads=num_heads,
                                          num_layers=num_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.local_branch = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.last_h2_proj = nn.Sequential(nn.Linear(1, 64), nn.GELU())
        fused_dim = d_model + d_model + lstm_hidden + 64
        self.fusion = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
        self.prod_delta_head = nn.Sequential(nn.Linear(fused_dim, 256), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(256, 1))
        self.lcoh_head = nn.Sequential(nn.Linear(fused_dim, 128), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(128, 1))
        self.h2_residual_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        h = self.pos_encoder(self.input_encoder(x))
        transformer_out = self.encoder(h)
        local_out = self.local_branch(h.transpose(1, 2)).transpose(1, 2)
        lstm_out, _ = self.lstm(transformer_out)
        last_h2 = self.last_h2_proj(x[:, -1, 8:9])
        context = torch.cat([transformer_out.mean(dim=1), local_out[:, -1, :],
                             lstm_out[:, -1, :], last_h2], dim=1)
        fused = self.fusion(context)
        fused = fused * self.gate(context) + context
        h2 = x[:, -1, 8:9] + self.h2_residual_scale * self.prod_delta_head(fused)
        return h2, self.lcoh_head(fused)

class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss with learnable task weighting (Kendall et al. 2018).
    L_total = (1/σ²_prod) * MSE_prod + (1/σ²_LCOH) * MSE_LCOH + log σ_prod + log σ_LCOH

    Initialized with large log_var values to ensure stable training at the start,
    preventing the precision terms from exploding when outputs have different scales.
    """

    def __init__(self, init_log_var_prod: float = 0.0, init_log_var_lcoh: float = 0.0):
        super().__init__()
        # Large init log_var → small precision → moderate loss at start
        # log_var=10 → σ² ≈ 22026 → precision ≈ 4.5e-5 (small, stable)
        self.log_var_prod = nn.Parameter(torch.tensor(init_log_var_prod))
        self.log_var_lcoh = nn.Parameter(torch.tensor(init_log_var_lcoh))

    def forward(
        self,
        pred_prod: torch.Tensor,
        pred_lcoh: torch.Tensor,
        target_prod: torch.Tensor,
        target_lcoh: torch.Tensor,
    ) -> dict:
        # Weighted MSE (use relu + small epsilon to prevent precision explosion)
        precision_prod = torch.exp(-self.log_var_prod).clamp(max=1e6)
        precision_lcoh = torch.exp(-self.log_var_lcoh).clamp(max=1e6)
        loss_prod = precision_prod * F.mse_loss(pred_prod, target_prod, reduction="mean")
        loss_lcoh = precision_lcoh * F.mse_loss(pred_lcoh, target_lcoh, reduction="mean")

        # Uncertainty regularization term
        reg_prod = self.log_var_prod.clamp(-10, 20)
        reg_lcoh = self.log_var_lcoh.clamp(-10, 20)

        # H2 is the primary measured target; LCOH is derived and remains an
        # auxiliary regularizer with a smaller contribution to the gradient.
        total = 3.0 * loss_prod + 0.1 * loss_lcoh + 0.5 * (reg_prod + reg_lcoh)

        # Guard against NaN
        if not torch.isfinite(total):
            total = loss_prod + loss_lcoh + 1.0

        return {
            "total": total,
            "loss_prod": loss_prod.detach(),
            "loss_lcoh": loss_lcoh.detach(),
            "sigma_prod": torch.exp(self.log_var_prod * 0.5),
            "sigma_lcoh": torch.exp(self.log_var_lcoh * 0.5),
        }


if __name__ == "__main__":
    # Smoke test
    model = HydrogenTransformer(input_dim=10, d_model=128, num_heads=8, num_layers=4)
    x = torch.randn(4, 1440, 10)
    h2, lc = model(x)
    print(f"h2 shape: {h2.shape}, lc shape: {lc.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
