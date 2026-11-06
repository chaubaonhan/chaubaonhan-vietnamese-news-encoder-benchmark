"""Fig 4 (revised): mean test-set metrics of the nine checkpoints, with the
value printed above each bar (Reviewer #1, Concern #3).

Values are the three-seed means from Table 2 of the manuscript.
Usage: python make_fig4_metrics.py OUTPUT_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

# Table 2 (mean over seeds 42, 22, 202)
MODELS = ["WikiBERT", "PhoBERT", "mBERT", "DistilBERT", "XLM-R",
          "CafeBERT", "viBERT", "vELECTRA", "ViSoBERT"]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2",
          "#72B7B2", "#FF9DA6", "#9D755D", "#BAB0AC"]
DATA = {  # accuracy, macro-P, macro-R, macro-F1, macro-AUC
    "WikiBERT":   [0.93016, 0.92377, 0.92264, 0.92302, 0.99635],
    "PhoBERT":    [0.94779, 0.94124, 0.94375, 0.94241, 0.99794],
    "mBERT":      [0.94270, 0.93707, 0.93660, 0.93674, 0.99750],
    "DistilBERT": [0.93793, 0.93145, 0.93213, 0.93167, 0.99713],
    "XLM-R":      [0.94488, 0.93907, 0.94011, 0.93953, 0.99770],
    "CafeBERT":   [0.94916, 0.94391, 0.94445, 0.94413, 0.99800],
    "viBERT":     [0.92903, 0.92226, 0.92185, 0.92190, 0.99629],
    "vELECTRA":   [0.93791, 0.93174, 0.93189, 0.93161, 0.99710],
    "ViSoBERT":   [0.92617, 0.91819, 0.92075, 0.91934, 0.99604],
}
METRICS = ["Accuracy", "Macro-precision", "Macro-recall", "Macro-F1", "Macro-AUC"]

vals = np.array([DATA[m] for m in MODELS]) * 100  # percent

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.5, 4.6), gridspec_kw={"width_ratios": [4, 1.15]})

n = len(MODELS)
w = 0.095


def fmt_half_up(frac, fmt):
    """Round the Table 2 value (in %) half-up, as in the manuscript text."""
    nd = int(fmt.split(".")[1][0])
    q = Decimal(str(frac)) * 100
    return str(q.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def draw(ax, metric_idx, ylim, fmt):
    for j, mi in enumerate(metric_idx):
        for i in range(n):
            x = j + (i - (n - 1) / 2) * w
            v = vals[i, mi]
            ax.bar(x, v, width=w, color=COLORS[i], edgecolor="black",
                   linewidth=0.4, label=MODELS[i] if j == 0 else None)
            ax.text(x, v + (ylim[1] - ylim[0]) * 0.006, fmt_half_up(DATA[MODELS[i]][mi], fmt),
                    rotation=90, ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(range(len(metric_idx)))
    ax.set_xticklabels([METRICS[k] for k in metric_idx], fontweight="bold")
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


draw(ax1, [0, 1, 2, 3], (91.0, 95.6), "{:.2f}")
ax1.set_ylabel("Score (%)", fontweight="bold")
ax1.set_title("(a) Accuracy and macro-averaged P / R / F1", fontsize=9)

draw(ax2, [4], (99.55, 99.85), "{:.3f}")
ax2.set_title("(b) Macro-AUC", fontsize=9)
ax2.set_xlim(-0.55, 0.55)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True,
           fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1))

fig.savefig(OUT / "Fig4.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig4.tif", dpi=400, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"})
print("saved", OUT / "Fig4.pdf", OUT / "Fig4.tif")
