"""Create a visual comparison between PT-MELT and the project Transformer."""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    err = pred - true
    mask = np.abs(true) > 1e-6
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[mask] / true[mask])) * 100),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot else float("nan"),
    }


def main(args):
    pt = pd.read_csv(args.ptmelt)
    tr = pd.read_csv(args.transformer)
    aligned = False
    key = ["experiment", "experiment_position"]
    if set(key).issubset(pt.columns) and set(key).issubset(tr.columns):
        pt["experiment_position"] = pd.to_numeric(pt["experiment_position"], errors="coerce")
        tr["experiment_position"] = pd.to_numeric(tr["experiment_position"], errors="coerce")
        pt = pt.drop_duplicates(key, keep="first")
        tr = tr.drop_duplicates(key, keep="first")
        merged = pt.merge(tr, on=key, how="inner", validate="one_to_one", suffixes=("_ptmelt", "_transformer"))
        if len(merged):
            pt_true = merged["h2_true_ptmelt"].to_numpy()
            pt_pred = merged["h2_pred_ptmelt"].to_numpy()
            tr_true = merged["h2_true_transformer"].to_numpy()
            tr_pred = merged["h2_pred_transformer"].to_numpy()
            aligned = True
        else:
            merged = None
    if not aligned:
        merged = None
        pt_true, pt_pred = pt["h2_true"].to_numpy(), pt["h2_pred"].to_numpy()
        tr_true, tr_pred = tr["h2_true"].to_numpy(), tr["h2_pred"].to_numpy()
    pt_m, tr_m = metrics(pt_pred, pt_true), metrics(tr_pred, tr_true)

    n = min(args.plot_points, len(pt_true), len(tr_true))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=150)
    axes[0, 0].plot(pt_true[:n], label="True", lw=1.2, color="#222222")
    axes[0, 0].plot(pt_pred[:n], label="PT-MELT", lw=1.0, color="#0072B2")
    axes[0, 0].set_title("PT-MELT: H2 prediction")
    axes[0, 0].set_ylabel("H2 flow (kg/h)")
    axes[0, 0].legend()

    axes[0, 1].plot(tr_true[:n], label="True", lw=1.2, color="#222222")
    axes[0, 1].plot(tr_pred[:n], label=args.reference_label, lw=1.0, color="#D55E00")
    axes[0, 1].set_title(f"{args.reference_label}: H2 prediction")
    axes[0, 1].set_ylabel("H2 flow (kg/h)")
    axes[0, 1].legend()

    axes[1, 0].scatter(pt_true, pt_pred, s=3, alpha=.15, color="#0072B2", label="PT-MELT")
    axes[1, 0].scatter(tr_true, tr_pred, s=8, alpha=.35, color="#D55E00", label=args.reference_label)
    lo = min(pt_true.min(), tr_true.min(), pt_pred.min(), tr_pred.min())
    hi = max(pt_true.max(), tr_true.max(), pt_pred.max(), tr_pred.max())
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1, 0].set_title("Prediction parity")
    axes[1, 0].set_xlabel("True H2 (kg/h)")
    axes[1, 0].set_ylabel("Predicted H2 (kg/h)")
    axes[1, 0].legend()

    axes[1, 1].bar([0, 1], [pt_m["mae"], tr_m["mae"]], color=["#0072B2", "#D55E00"], width=.6)
    axes[1, 1].set_xticks([0, 1], ["PT-MELT", args.reference_label])
    axes[1, 1].set_ylabel("MAE (kg/h)")
    axes[1, 1].set_title("H2 MAE comparison")
    for i, value in enumerate([pt_m["mae"], tr_m["mae"]]):
        axes[1, 1].text(i, value, f"{value:.2f}", ha="center", va="bottom")

    fig.suptitle("Hydrogen production model comparison", fontsize=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    report = {"ptmelt": pt_m, "transformer": tr_m,
              "ptmelt_samples": len(pt), "transformer_samples": len(tr),
              "aligned_samples": int(len(pt_true)) if aligned else None,
              "aligned_by": "experiment + experiment_position" if aligned else None,
              "note": "GT curves are identical when aligned=True. Otherwise the files were generated before metadata support was added."}
    report_path = os.path.splitext(args.output)[0] + ".json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved_plot={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptmelt", default="outputs/ptmelt_aligned_predictions.csv")
    parser.add_argument("--transformer", default="outputs/transformer_test_predictions.csv")
    parser.add_argument("--output", default="outputs/figures/model_comparison.png")
    parser.add_argument("--plot-points", type=int, default=1500)
    parser.add_argument("--reference-label", default="HydroFormerLSTM")
    main(parser.parse_args())



