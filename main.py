"""
Main entry point for the Green Hydrogen Production deep learning project.

Usage:
    python main.py                     # Run full pipeline
    python main.py --model transformer # Run single model
    python main.py --epochs 100        # Custom epochs
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure src package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data_loader import (
    load_and_preprocess, build_dataloaders,
    INPUT_KEYS, OUTPUT_KEYS,
)
from src.models.transformer import HydrogenTransformer, HydrogenTransformerLSTM, HydroFormerLSTM
from src.models.baselines import HydrogenLSTM, HydrogenGRU, Hydrogen1DCNN, HydrogenXGBoost
from src.trainer import HydrogenTrainer, evaluate, save_prediction_artifacts


# ────────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ────────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────────

CONFIG = dict(
    # Data
    raw_csv       = os.path.join(BASE_DIR, "data", "raw", "combined_wind_experiments.csv"),
    # Window parameters for 1 Hz data:
    #   lookback=360  →  6 min history  (enough to capture electrolyzer dynamics)
    #   horizon=60    →  1 min forecast
    #   stride=30     →  30-s stride  (sparse enough to avoid excessive overlap)
    lookback      = 180,
    horizon       = 30,
    stride        = 15,
    batch_size    = 32,

    # Transformer / hybrid
    d_model       = 128,
    hybrid_d_model = 192,
    hybrid_layers = 4,
    hybrid_lstm_hidden = 256,
    hybrid_dim_ff = 768,
    num_heads     = 8,
    num_layers    = 4,
    dim_ff        = 768,
    dropout       = 0.15,

    # Training
    lr            = 1e-4,
    weight_decay  = 0.01,
    warmup_epochs = 10,
    max_epochs    = 200,
    patience      = 30,
    max_grad_norm = 1.0,
    use_amp       = False,   # AMP requires CUDA; disabled for CPU

    # Baselines
    lstm_hidden   = 128,
    gru_hidden    = 128,
    cnn_hidden    = 64,

    # Misc
    seed          = 42,
    output_dir    = os.path.join(BASE_DIR, "outputs"),
)


# ────────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ────────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    scaler_y,
    device,
    config: dict,
):
    """Train a single model and return final test metrics."""

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")

    # Build model
    if model_name in ("hydroformer", "hybrid"):
        model = HydroFormerLSTM(
            input_dim=len(INPUT_KEYS),
            d_model=config["hybrid_d_model"],
            num_heads=config["num_heads"],
            num_layers=config["hybrid_layers"],
            lstm_hidden=config["hybrid_lstm_hidden"],
            dim_feedforward=config["hybrid_dim_ff"],
            dropout=config["dropout"],
        )
    elif model_name == "transformer":
        model = HydrogenTransformer(
            input_dim=len(INPUT_KEYS),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_ff"],
            dropout=config["dropout"],
        )
    elif model_name == "lstm":
        model = HydrogenLSTM(
            input_dim=len(INPUT_KEYS),
            hidden_dim=config["lstm_hidden"],
            num_layers=2,
            dropout=config["dropout"],
        )
    elif model_name == "gru":
        model = HydrogenGRU(
            input_dim=len(INPUT_KEYS),
            hidden_dim=config["gru_hidden"],
            num_layers=2,
            dropout=config["dropout"],
        )
    elif model_name == "cnn":
        model = Hydrogen1DCNN(
            input_dim=len(INPUT_KEYS),
            hidden_channels=config["cnn_hidden"],
            dropout=config["dropout"],
        )
    elif model_name == "xgboost":
        model = HydrogenXGBoost()
        # XGBoost uses a separate training path
        trainer = HydrogenTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            log_dir=config["output_dir"],
            model_name=model_name,
            use_amp=False,
        )
        trainer.train(scaler_y=scaler_y)
        # Evaluate
        test_metrics = evaluate(model, test_loader, device, scaler_y, is_xgb=True)
        save_prediction_artifacts(model, test_loader, device, scaler_y,
                                  config["output_dir"], model_name,
                                  stride=config["stride"], lookback=config["lookback"],
                                  horizon=config["horizon"], is_xgb=True)
        return test_metrics, trainer.history

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    trainer = HydrogenTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        max_grad_norm=config["max_grad_norm"],
        use_amp=config["use_amp"],
        log_dir=config["output_dir"],
        model_name=model_name,
    )

    start_time = time.time()
    history = trainer.train(scaler_y=scaler_y)
    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f}s")

    # Save checkpoint
    os.makedirs(config["output_dir"], exist_ok=True)
    ckpt_path = os.path.join(config["output_dir"], f"{model_name}_best.pt")
    trainer.save(ckpt_path)

    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device, scaler_y)
    pred_csv, pred_png = save_prediction_artifacts(
        model, test_loader, device, scaler_y, config["output_dir"], model_name,
        stride=config["stride"], lookback=config["lookback"], horizon=config["horizon"],
    )
    print(f"  Prediction CSV: {pred_csv}")
    if pred_png:
        print(f"  Prediction plot: {pred_png}")
    print(f"\n  Test Results ({model_name}):")
    for target, metrics in test_metrics.items():
        print(f"    {target}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
              f"MAPE={metrics['mape']:.2f}%, R2={metrics['r2']:.4f}")

    return test_metrics, history


# ────────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────────

def main(args):
    config = {**CONFIG}
    if args.epochs:
        config["max_epochs"] = args.epochs
    if args.model:
        config["models"] = [args.model]
    else:
        config["models"] = ["hydroformer", "xgboost", "cnn", "gru", "lstm", "transformer"]

    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # ── Step 1: Load and preprocess data ───────────────────────────────────────
    df = load_and_preprocess(config["raw_csv"])

    # ── Step 2: Build dataloaders ─────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler_tuple = build_dataloaders(
        df,
        lookback=config["lookback"],
        horizon=config["horizon"],
        stride=config["stride"],
        batch_size=config["batch_size"],
    )
    scaler_X, scaler_y = scaler_tuple  # unpack tuple of scalers
    print(f"\nDataset sizes: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    # ── Step 3: Run experiments ────────────────────────────────────────────────
    results = {}
    for model_name in config["models"]:
        test_metrics, history = run_experiment(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scaler_y=scaler_y,
            device=device,
            config=config,
        )
        results[model_name] = {
            "test_metrics": {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in test_metrics.items()},
            "history": {k: [float(v_) for v_ in v] for k, v in history.items()
                        if isinstance(v, list)},
        }

    # ── Step 4: Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':<15} {'H2 MAE':>10} {'H2 RMSE':>10} {'H2 MAPE':>10} {'LCOH MAE':>10} {'LCOH RMSE':>10}"
    print(header)
    print("-" * 70)
    for name, res in results.items():
        m = res["test_metrics"]
        print(f"{name:<15} "
              f"{m['h2_production']['mae']:>10.4f} "
              f"{m['h2_production']['rmse']:>10.4f} "
              f"{m['h2_production']['mape']:>10.2f} "
              f"{m['lcost']['mae']:>10.4f} "
              f"{m['lcost']['rmse']:>10.4f}")

    # Save results
    results_path = os.path.join(config["output_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print("Project complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Green Hydrogen Production DL Project")
    parser.add_argument("--model",  type=str,  default=None,
                        help="Model to run: hydroformer, transformer, lstm, gru, cnn, xgboost")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max training epochs")
    args = parser.parse_args()
    main(args)
