"""
Training and evaluation pipeline for the green hydrogen production project.
Implements the full training protocol from methodology §II.C:
  - AdamW optimizer (lr=1e-4, weight_decay=0.01)
  - Cosine annealing scheduler with 10-epoch warm-up
  - Gradient clipping (max_norm=1.0)
  - Mixed-precision training (FP16)
  - Early stopping (patience=30 epochs)
  - Evaluation: MAE, RMSE, MAPE, R²
"""

import os
import math
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .data_loader import INPUT_KEYS, OUTPUT_KEYS


# ────────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE (%), and R² for a single output."""
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()
    mae  = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    # MAPE: avoid division by zero
    mask = np.abs(target) > 1e-6
    mape = np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100 if mask.any() else float("nan")
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def evaluate(model, loader, device, scaler_y=None, is_xgb=False):
    """Run full evaluation on a DataLoader and return per-target metrics."""
    if not is_xgb:
        model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            if is_xgb:
                # XGBoost: numpy interface
                x_np = batch_x.numpy().reshape(batch_x.size(0), -1)
                pred = model.predict(x_np)
            else:
                x = batch_x.to(device)
                p_h2, p_lc = model(x)
                pred = torch.cat([p_h2, p_lc], dim=1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(batch_y.numpy())

    preds   = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Inverse-transform to original scale
    if scaler_y is not None:
        preds   = scaler_y.inverse_transform(preds)
        targets = scaler_y.inverse_transform(targets)

    h2_metrics  = compute_metrics(preds[:, 0], targets[:, 0])
    lc_metrics  = compute_metrics(preds[:, 1], targets[:, 1])

    return {"h2_production": h2_metrics, "lcost": lc_metrics}


def collect_predictions(model, loader, device, scaler_y=None, is_xgb=False):
    """Return inverse-scaled predictions and targets for a loader."""
    if not is_xgb:
        model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            if is_xgb:
                x_np = batch_x.numpy().reshape(batch_x.size(0), -1)
                pred = model.predict(x_np)
            else:
                x = batch_x.to(device)
                p_h2, p_lc = model(x)
                pred = torch.cat([p_h2, p_lc], dim=1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(batch_y.numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds)
        targets = scaler_y.inverse_transform(targets)
    return preds, targets


def save_prediction_artifacts(model, loader, device, scaler_y, output_dir, model_name, stride=30, lookback=360, horizon=60, is_xgb=False):
    """Save test predictions as CSV and H2/LCOH comparison plot."""
    preds, targets = collect_predictions(model, loader, device, scaler_y, is_xgb=is_xgb)
    print(f"  H2 target range: {targets[:, 0].min():.3f} .. {targets[:, 0].max():.3f}, mean={targets[:, 0].mean():.3f}")
    print(f"  H2 prediction range: {preds[:, 0].min():.3f} .. {preds[:, 0].max():.3f}, mean={preds[:, 0].mean():.3f}")
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    sample_index = np.arange(len(preds))
    elapsed_seconds = lookback + horizon - 1 + sample_index * stride
    csv_path = os.path.join(output_dir, f"{model_name}_test_predictions.csv")
    metadata = getattr(loader.dataset, "window_metadata", [{} for _ in range(len(preds))])
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_index,elapsed_seconds,experiment,experiment_position,target_timestamp,h2_true,h2_pred,lcoh_true,lcoh_pred\n")
        for i, t in enumerate(elapsed_seconds):
            meta = metadata[i] if i < len(metadata) else {}
            experiment = str(meta.get("experiment", "unknown")).replace(",", "_")
            position = int(meta.get("experiment_position", -1))
            timestamp = str(meta.get("target_timestamp", ""))
            f.write(f"{i},{int(t)},{experiment},{position},{timestamp},{targets[i, 0]:.8f},{preds[i, 0]:.8f},{targets[i, 1]:.8f},{preds[i, 1]:.8f}\n")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(elapsed_seconds, targets[:, 0], label="H2 true", linewidth=1.6)
        axes[0].plot(elapsed_seconds, preds[:, 0], label="H2 predicted", linewidth=1.2)
        axes[0].set_ylabel("H2 (kg/h)")
        axes[0].set_title(f"{model_name.upper()} Test set: H2 production prediction vs truth")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(elapsed_seconds, targets[:, 1], label="LCOH true", linewidth=1.6)
        axes[1].plot(elapsed_seconds, preds[:, 1], label="LCOH predicted", linewidth=1.2)
        axes[1].set_xlabel("Elapsed time in test sequence (s)")
        axes[1].set_ylabel("LCOH ($/kg)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        png_path = os.path.join(output_dir, "figures", f"{model_name}_test_predictions.png")
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
    except ImportError:
        png_path = None
    return csv_path, png_path

# ────────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler with warm-up + cosine annealing
# ────────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lr      = optimizer.param_groups[0]["lr"]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ────────────────────────────────────────────────────────────────────────────────
# Main trainer
# ────────────────────────────────────────────────────────────────────────────────

class HydrogenTrainer:
    """
    Unified trainer compatible with PyTorch models and XGBoost.
    Supports mixed-precision training, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
        max_epochs: int = 500,
        patience: int = 30,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        log_dir: str = None,
        model_name: str = "model",
    ):
        self.model       = model
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.device      = device
        self.max_epochs  = max_epochs
        self.patience    = patience
        self.max_grad_norm = max_grad_norm
        self.use_amp     = use_amp
        self.log_dir     = log_dir
        self.model_name  = model_name
        self.is_xgb      = "XGBoost" in type(model).__name__

        if not self.is_xgb:
            self.model = self.model.to(device)

            # Learn task uncertainty jointly with the model weights.
            from .models.transformer import MultiTaskLoss
            self.criterion = MultiTaskLoss().to(device)
            self.optimizer = optim.AdamW(
                list(self.model.parameters()) + list(self.criterion.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            self.scheduler = WarmupCosineScheduler(
                self.optimizer, warmup_epochs=warmup_epochs, total_epochs=max_epochs
            )
            self.amp_scaler = GradScaler() if use_amp else None

        self.best_val_loss = float("inf")
        self.best_state    = None
        self.wait = 0
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_h2_mae": [],
            "train_lc_mae": [],
            "val_loss": [],
            "val_h2_mae": [],
            "val_lc_mae": [],
        }

    def _pytorch_step(self, batch_x, batch_y):
        self.model.train()
        x = batch_x.to(self.device)
        targets = batch_y.to(self.device)
        if self.use_amp:
            with autocast():
                p_h2, p_lc = self.model(x)
                loss_dict = self.criterion(p_h2, p_lc, targets[:, 0:1], targets[:, 1:2])
            total_loss = loss_dict["total"]
            self.amp_scaler.scale(total_loss).backward()
            self.amp_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            p_h2, p_lc = self.model(x)
            loss_dict = self.criterion(p_h2, p_lc, targets[:, 0:1], targets[:, 1:2])
            total_loss = loss_dict["total"]
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss_dict

    def _train_epoch(self):
        epoch_losses = []
        for batch_x, batch_y in self.train_loader:
            if self.is_xgb:
                raise RuntimeError("XGBoost uses fit(), not step-by-step training")
            epoch_losses.append(self._pytorch_step(batch_x, batch_y)["total"].item())
        return np.mean(epoch_losses)

    def train(self, scaler_y=None):
        if self.is_xgb:
            return self._train_xgb(scaler_y)
        for epoch in range(1, self.max_epochs + 1):
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_loss = self._train_epoch()
            train_metrics = evaluate(self.model, self.train_loader, self.device, scaler_y)
            val_metrics = evaluate(self.model, self.val_loader, self.device, scaler_y)
            train_h2_mae = train_metrics["h2_production"]["mae"]
            train_lc_mae = train_metrics["lcost"]["mae"]
            val_h2_mae = val_metrics["h2_production"]["mae"]
            val_lc_mae = val_metrics["lcost"]["mae"]
            # H2 production is the primary operational target. H2 MAE and LCOH
            # MAE have different physical units, so summing them would make
            # checkpoint selection depend on an arbitrary unit conversion.
            val_loss = val_h2_mae
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_h2_mae"].append(train_h2_mae)
            self.history["train_lc_mae"].append(train_lc_mae)
            self.history["val_loss"].append(val_loss)
            self.history["val_h2_mae"].append(val_h2_mae)
            self.history["val_lc_mae"].append(val_lc_mae)
            print(f"[Epoch {epoch:3d}] lr={current_lr:.2e} | train_loss={train_loss:.4f} | "
                  f"train_h2_MAE={train_h2_mae:.4f} | val_h2_MAE={val_h2_mae:.4f} | "
                  f"val_lc_MAE={val_lc_mae:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch}")
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def _train_xgb(self, scaler_y):
        X_train = np.concatenate([b[0].numpy().reshape(b[0].size(0), -1) for b in self.train_loader], axis=0)
        y_train = np.concatenate([b[1].numpy() for b in self.train_loader], axis=0)
        # Keep XGBoost targets in standardized space; evaluate inverse-scales once.
        self.model.fit(X_train, y_train)
        metrics = evaluate(self.model, self.val_loader, self.device, scaler_y, is_xgb=True)
        self.history["val_h2_mae"] = [metrics["h2_production"]["mae"]]
        self.history["val_lc_mae"] = [metrics["lcost"]["mae"]]
        return self.history

    def save(self, path: str):
        if self.is_xgb:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
        else:
            torch.save({
                "model_state": self.best_state,
                "criterion_state": self.criterion.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
            }, path)

    @staticmethod
    def load(path: str, model_class=None, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if model_class is not None:
            model = model_class()
            model.load_state_dict(ckpt["model_state"])
            return model, ckpt["history"]
        return ckpt


if __name__ == "__main__":
    print("trainer.py loaded OK")
