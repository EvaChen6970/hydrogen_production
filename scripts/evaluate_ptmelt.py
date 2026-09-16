"""Evaluate the published NLR PT-MELT LSTM on an NLR CSV file.

This evaluates both one-step-ahead and 30-second-ahead predictions. The
published model was trained for one-step prediction, so the one-step number is
the fair comparison to its original task.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ptmelt import load_ptmelt_model


FEATURES = [
    "H2E_f_NLR_CurrentCmd", "H2E_f_PT264_Pressure", "H2E_f_PT307_Pressure",
    "H2E_f_PT312_Pressure", "H2E_f_PT604_Pressure", "H2E_f_RS209_Resistivity",
    "H2E_f_RS507_Resistivity", "H2E_f_TE218_Temp", "H2E_f_TE219_Temp",
    "H2E_f_TE338_Temp", "H2E_f_TE601_Temp", "H2E_f_TT645_Temp",
    "H2E_f_TT646_Temp", "H2E_f_TT647_Temp", "H2E_f_TT648_Temp",
]
MEAN = np.array([52.0813833463, 2.8733982348, 28.9957708187, 24.3821938514,
                 4.4333828455, 6.6180255662, 10.7999670366, 57.2198698321,
                 57.8621499172, 10.8703047887, 28.1882504658, 12.7266522510,
                 10.6385243335, 10.9705712897, 16.7990789397], dtype=np.float32)
SCALE = np.array([27.0162048489, .1428419883, .4128271503, 2.8980042338,
                  .0846463460, .3347519228, 4.3784033552, 1.1730608722,
                  1.1563768046, .9732472206, 6.0748688831, 7.2756356283,
                  4.4313754092, 3.5504208358, 14.7519059835], dtype=np.float32)
Y_MEAN, Y_SCALE = 11.3110796462, 6.2371842488


def metrics(pred, true):
    err = pred - true
    mask = np.abs(true) > 1e-6
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[mask] / true[mask])) * 100),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def main(args):
    df = pd.read_csv(args.csv)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    target_col = "IVAL_f_FM011_Flow" if "IVAL_f_FM011_Flow" in df else "IVAL_f_FM011_Flow (kg/hr)"
    if "H2E_f_TT648_Temp" not in df:
        # The simulated-wind release exposes TT641 where the historical release
        # exposes TT648; both are the final temperature channel for this model.
        df["H2E_f_TT648_Temp"] = df["H2E_f_TT641_Temp"]
    data = df[["timestamp"] + FEATURES + [target_col, "Experiment"]].copy()
    data["experiment_position"] = data.groupby("Experiment", sort=False).cumcount()
    for col in FEATURES + [target_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[FEATURES + [target_col]] = data[FEATURES + [target_col]].groupby(data["Experiment"], sort=False).ffill()
    data = data.dropna(subset=FEATURES + [target_col])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_ptmelt_model(args.model, device)
    all_pred, all_true = {1: [], args.horizon: []}, {1: [], args.horizon: []}
    metadata = []
    with torch.no_grad():
        for _, group in data.groupby("Experiment", sort=False):
            x = group[FEATURES].to_numpy(np.float32)
            y = group[target_col].to_numpy(np.float32)
            for t in range(args.seq_length, len(group) - args.horizon + 1):
                window = (x[t - args.seq_length:t] - MEAN) / SCALE
                pred, _ = model(torch.from_numpy(window[None]).to(device))
                value = float(pred.squeeze().cpu()) * Y_SCALE + Y_MEAN
                for horizon in (1, args.horizon):
                    if t + horizon - 1 < len(y):
                        all_pred[horizon].append(value)
                        all_true[horizon].append(float(y[t + horizon - 1]))
                        if horizon == args.horizon:
                            metadata.append({
                                "experiment": str(group.iloc[t + horizon - 1]["Experiment"]),
                                "experiment_position": int(group.iloc[t + horizon - 1]["experiment_position"]),
                                "target_timestamp": str(group.iloc[t + horizon - 1]["timestamp"]),
                            })
    for horizon in (1, args.horizon):
        p, y = np.array(all_pred[horizon]), np.array(all_true[horizon])
        print(f"PT-MELT horizon={horizon}s n={len(y)} metrics={metrics(p, y)}")
    out = pd.DataFrame({"experiment": [m["experiment"] for m in metadata],
                        "experiment_position": [m["experiment_position"] for m in metadata],
                        "target_timestamp": [m["target_timestamp"] for m in metadata],
                        "h2_true": all_true[args.horizon], "h2_pred": all_pred[args.horizon]})
    if args.reference and os.path.exists(args.reference):
        ref = pd.read_csv(args.reference)
        required = {"experiment", "experiment_position"}
        if required.issubset(ref.columns):
            ref_keys = set(zip(ref["experiment"].astype(str),
                               pd.to_numeric(ref["experiment_position"], errors="coerce")))
            out_keys = list(zip(out["experiment"].astype(str), out["experiment_position"]))
            out = out[[k in ref_keys for k in out_keys]].copy()
            print(f"aligned_to={args.reference} aligned_samples={len(out)}")
    out.to_csv(args.output, index=False)
    print(f"saved={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/combined_wind_experiments.csv")
    parser.add_argument("--model", default="outputs/h2e_lstm_model_60s_attn_15inputs.safetensors")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--seq-length", type=int, default=60)
    parser.add_argument("--output", default="outputs/ptmelt_aligned_predictions.csv")
    parser.add_argument("--reference", default="outputs/hydroformer_test_predictions.csv", help="CSV whose experiment/window keys define the comparison set; pass empty string to disable")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
