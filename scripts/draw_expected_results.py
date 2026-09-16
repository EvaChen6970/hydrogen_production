"""Generate an illustrative figure summarizing what this project aims to achieve.

The figure shows the EXPECTED behavior of the system, not the actual run results
(which were poor due to tiny training set, as discussed in README §9.1).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

os.makedirs("outputs/figures", exist_ok=True)
np.random.seed(42)


# ── Color palette ────────────────────────────────────────────────────────────
C_INPUT  = "#4C72B0"   # blue
C_MODEL  = "#55A868"   # green
C_H2     = "#C44E52"   # red
C_LCOH   = "#8172B2"   # purple
C_ARROW  = "#444444"
C_PANEL  = "#F2F2F2"


def draw_box(ax, xy, w, h, text, fc, fontsize=10, weight="normal"):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="#333333", facecolor=fc, alpha=0.85,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#222222", wrap=True)


def draw_arrow(ax, start, end, color=C_ARROW, lw=1.5):
    arrow = FancyArrowPatch(start, end, arrowstyle="->,head_length=0.4,head_width=0.3",
                            color=color, linewidth=lw, mutation_scale=15)
    ax.add_patch(arrow)


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.0, 1.0], hspace=0.45, wspace=0.35)

# ============================================================
# Panel 1: System Architecture (top, full width)
# ============================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 5)
ax1.axis("off")
ax1.set_title("Expected Pipeline — Input → Model → Dual Outputs",
              fontsize=14, weight="bold", pad=8)

# Input window (T=360, D=7)
draw_box(ax1, (0.3, 2.5), 2.0, 1.6,
         "Raw CSV\n28,800 rows × 1 Hz\n(16 hours total)",
         C_PANEL, fontsize=9)
draw_box(ax1, (0.3, 0.7), 2.0, 1.2,
         "Sliding Window\nT=360, H=60, stride=30\n→ 921 windows",
         C_INPUT, fontsize=9)

# Feature extraction
draw_box(ax1, (3.0, 1.5), 2.2, 1.8,
         "Feature Engineering\n7 channels:\n• wind_speed\n• turbulence\n• dc_current\n• ramp_rate\n• temp / pressure\n• power",
         "#E8E1F5", fontsize=8.5)

# Transformer encoder
draw_box(ax1, (6.0, 2.2), 2.4, 1.4,
         "Transformer Encoder\n3 layers × 4 heads\nd_model=128",
         C_MODEL, fontsize=10, weight="bold")

# Dual heads
draw_box(ax1, (9.2, 3.0), 1.9, 1.0,
         "Production Head\nLinear(128→1)",
         "#F4DCDC", fontsize=9)
draw_box(ax1, (9.2, 1.2), 1.9, 1.0,
         "LCOH Head\nLinear(128→1)\n+ skip conn.",
         "#E1DCF4", fontsize=9)

# Outputs
draw_box(ax1, (11.6, 3.0), 2.2, 1.0,
         "Ĥ₂ (kg/h)\nground truth:\nIVAL_f_FM011_Flow",
         C_H2, fontsize=8.5, weight="bold")
draw_box(ax1, (11.6, 1.2), 2.2, 1.0,
         "LCOH ($/kg)\ncomputed from\nefficiency",
         C_LCOH, fontsize=8.5, weight="bold")

# Arrows
draw_arrow(ax1, (2.3, 3.3), (3.0, 2.6))
draw_arrow(ax1, (2.3, 1.3), (3.0, 2.0))
draw_arrow(ax1, (5.2, 2.4), (6.0, 2.9))
draw_arrow(ax1, (8.4, 3.2), (9.2, 3.5))
draw_arrow(ax1, (8.4, 2.6), (9.2, 1.7))
draw_arrow(ax1, (11.1, 3.5), (11.6, 3.5))
draw_arrow(ax1, (11.1, 1.7), (11.6, 1.7))


# ============================================================
# Panel 2: Input time-series example (top-right)
# ============================================================
ax2 = fig.add_subplot(gs[1, 0])
t = np.linspace(0, 6, 360)  # 6 minutes
ws = 5 + 3 * np.sin(t) + 0.4 * np.random.randn(360)
ti = 8 + 2 * np.abs(np.sin(t * 1.3)) + 0.5 * np.random.randn(360)
cur = 1800 + 400 * np.sin(t * 0.8) + 30 * np.random.randn(360)
ax2.plot(t, ws, color="#4C72B0", lw=1.3, label="wind_speed")
ax2.plot(t, ti + 5, color="#55A868", lw=1.3, label="turbulence")
ax2.plot(t, (cur - 1500) / 5, color="#C44E52", lw=1.3, label="dc_current (scaled)")
ax2.set_title("Input Window  (T=360 s, 6 min)", fontsize=11, weight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Channel (arbitrary scale)")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)


# ============================================================
# Panel 3: Expected H₂ prediction vs ground truth (top-middle)
# ============================================================
ax3 = fig.add_subplot(gs[1, 1])
t_h = np.linspace(0, 60, 60)
# Realistic H₂ production: oscillating 5–18 kg/h with some noise
gt_h2 = 12 + 4 * np.sin(t_h / 8) + 1.5 * np.sin(t_h / 3) + 0.5 * np.random.randn(60)
# Predicted H₂: tracks GT with small error
pred_h2 = gt_h2 + 0.6 * np.random.randn(60)
ax3.plot(t_h, gt_h2, color="black", lw=2.0, label="Ground truth (GT)")
ax3.plot(t_h, pred_h2, color=C_H2, lw=1.8, linestyle="--", label="Transformer forecast")
ax3.fill_between(t_h, gt_h2 - 1.0, gt_h2 + 1.0, color=C_H2, alpha=0.15, label="±1 kg/h band")
ax3.set_title("Expected H₂ Production Forecast  (next 60 s)", fontsize=11, weight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("H₂ (kg/h)")
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, alpha=0.3)


# ============================================================
# Panel 4: Expected LCOH prediction (top-right)
# ============================================================
ax4 = fig.add_subplot(gs[1, 2])
# LCOH inversely proportional to H₂ + slow drift
gt_lc = 90 + 30 / (gt_h2 / 12) + 2 * np.sin(t_h / 12) + 1.0 * np.random.randn(60)
pred_lc = gt_lc + 1.8 * np.random.randn(60)
ax4.plot(t_h, gt_lc, color="black", lw=2.0, label="GT (computed)")
ax4.plot(t_h, pred_lc, color=C_LCOH, lw=1.8, linestyle="--", label="Transformer forecast")
ax4.fill_between(t_h, gt_lc - 3.5, gt_lc + 3.5, color=C_LCOH, alpha=0.15, label="±$3.5/kg band")
ax4.set_title("Expected LCOH Forecast  (next 60 s)", fontsize=11, weight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("LCOH ($/kg)")
ax4.legend(loc="upper right", fontsize=8)
ax4.grid(True, alpha=0.3)


# ============================================================
# Panel 5: Expected model comparison (bottom-left)
# ============================================================
ax5 = fig.add_subplot(gs[2, 0])
models = ["XGBoost", "1D-CNN", "GRU", "LSTM", "Transformer"]
# Hypothetical MAE on H₂ with sufficient data (kg/h)
mae_h2 = [3.2, 2.5, 2.1, 2.0, 1.7]
# Hypothetical MAE on LCOH ($/kg)
mae_lc = [9.0, 7.5, 6.8, 6.5, 5.8]

x = np.arange(len(models))
w = 0.35
b1 = ax5.bar(x - w/2, mae_h2, w, label="H₂ MAE (kg/h)", color=C_H2, alpha=0.85)
b2 = ax5.bar(x + w/2, mae_lc, w, label="LCOH MAE ($/kg)", color=C_LCOH, alpha=0.85)
ax5.set_xticks(x)
ax5.set_xticklabels(models)
ax5.set_ylabel("MAE")
ax5.set_title("Expected Model Comparison  (with sufficient data)", fontsize=11, weight="bold")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")
for b in b1: ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                      f"{b.get_height():.1f}", ha="center", fontsize=8)
for b in b2: ax5.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                      f"{b.get_height():.1f}", ha="center", fontsize=8)


# ============================================================
# Panel 6: Expected training curves (bottom-middle)
# ============================================================
ax6 = fig.add_subplot(gs[2, 1])
epoch = np.arange(1, 101)
train_loss = 8.5 * np.exp(-epoch / 25) + 0.5 + 0.05 * np.random.randn(100)
val_loss   = 8.5 * np.exp(-epoch / 18) + 0.9 + 0.08 * np.random.randn(100)
val_loss[60:] += (epoch[60:] - 60) * 0.01  # slight overfit tail

ax6.plot(epoch, train_loss, color=C_MODEL, lw=2, label="train_loss")
ax6.plot(epoch, val_loss, color="#E08E45", lw=2, label="val_loss")
ax6.axvline(85, color="gray", linestyle=":", lw=1)
ax6.text(85, 6.5, " early stop", color="gray", fontsize=8, va="bottom")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Loss")
ax6.set_title("Expected Training Curve  (proper convergence)", fontsize=11, weight="bold")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)


# ============================================================
# Panel 7: Expected metric summary table (bottom-right)
# ============================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
ax7.set_title("Expected Final Metrics  (test set)", fontsize=11, weight="bold", pad=8)

table_data = [
    ["Metric", "H₂", "LCOH"],
    ["MAE",  "~1.7 kg/h",   "~5.8 $/kg"],
    ["RMSE", "~2.5 kg/h",   "~7.5 $/kg"],
    ["MAPE", "~25%*",       "~10%"],
    ["R²",   "0.80–0.90",   "0.70–0.85"],
]
tbl = ax7.table(cellText=table_data, loc="center", cellLoc="center",
                colWidths=[0.30, 0.35, 0.35])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 2.0)
# Header style
for j in range(3):
    tbl[(0, j)].set_facecolor("#404040")
    tbl[(0, j)].set_text_props(color="white", weight="bold")
# Color code by target
for i in range(1, 5):
    tbl[(i, 1)].set_facecolor("#FBE4E4")
    tbl[(i, 2)].set_facecolor("#EBE4F4")

ax7.text(0.5, 0.05,
         "* MAPE inflated by near-zero H₂ samples; see README §6",
         ha="center", fontsize=8, style="italic", color="gray",
         transform=ax7.transAxes)


# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "outputs/figures/expected_results.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")
plt.close()
