"""Analyses that can be done exactly from the authors' own released artefacts.

Inputs
  results/confusion_matrices.json - test-set confusion matrices (counts) of all 27
              runs (Figs 6-8 of the paper); row sums equal the test-set class totals.
  TCC       - seed-specific TCC = train_duration + predict_duration
              (results/tcc_per_run.csv); PhoBERT: Table 2 mean in every panel.
Outputs (figures/generated): Fig5.pdf/.tif, per_class_f1.csv/.tex, cafe_vs_phobert.csv, report.txt
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "generated"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 22, 202]
ORDER = ["WikiBERT", "PhoBERT", "mBERT", "DistilBERT", "XLM-R", "CafeBERT", "viBERT", "vELECTRA", "ViSoBERT"]
NAMES = ["Education", "Entertainment", "Business", "Law", "Health", "World", "Sports",
         "Politics", "Automotive", "Lifestyle"]
COLORS = {"WikiBERT": "#1f77b4", "PhoBERT": "#ff7f0e", "mBERT": "#2ca02c", "DistilBERT": "#d62728",
          "XLM-R": "#9467bd", "CafeBERT": "#8c564b", "viBERT": "#e377c2", "vELECTRA": "#17becf",
          "ViSoBERT": "#bcbd22"}  # colours of the original Fig 5
MARK = {"WikiBERT": "o", "PhoBERT": "s", "mBERT": "^", "DistilBERT": "D", "XLM-R": "P",
        "CafeBERT": "X", "viBERT": "v", "vELECTRA": "<", "ViSoBERT": ">"}
LEG = {"PhoBERT": "PhoBERT-base", "XLM-R": "XLM-R-base", "vELECTRA": "vELECTRA-base"}

# released JSON: train_duration + predict_duration
JSON_TCC = {
    "WikiBERT": {42: 3036.116060256958 + 36.86030101776123, 22: 3039.995393037796 + 36.81377840042114,
                 202: 3044.508507490158 + 36.94228148460388},
    "mBERT": {42: 3143.1457874774933 + 37.317203521728516, 22: 3143.6796958446503 + 37.31348967552185,
              202: 3145.9938263893127 + 37.324496269226074},
    "DistilBERT": {42: 1537.0417258739471 + 19.210743188858032, 22: 1534.7832238674164 + 19.188753128051758,
                   202: 1535.3672459125519 + 19.011032342910767},
    "XLM-R": {42: 3060.7024145126343 + 36.558231830596924, 22: 3062.594987154007 + 36.57721996307373,
              202: 3064.1070318222046 + 36.62421226501465},
    "viBERT": {42: 3025.519765138626 + 38.01546096801758, 22: 3027.1372792720795 + 38.019421100616455,
               202: 3028.560325860977 + 38.035245180130005},
    "vELECTRA": {42: 3152.9810194969177 + 37.877885580062866, 22: 3154.5947160720825 + 37.65400838851929,
                 202: 3156.718153476715 + 37.95112061500549},
    "ViSoBERT": {42: 3172.4486666336 + 39.7513837779866, 22: 3171.5187845230103 + 39.67577791213989,
                 202: 3172.36838555336 + 39.70138359069824},
}
JSON_TCC["CafeBERT"] = {  # Drive: CafeBERT_Results/results_cafebert/all_seeds_metrics_cafebert.json
    42: 9019.929540872574 + 110.27680015563965, 22: 9366.849911928177 + 109.43493366241455,
    202: 9291.16268658638 + 108.86574959754944}
# PhoBERT: no seed-level timing that reproduces Table 2 exists -> the Table 2 mean
# (3,360.89 s, SD 4.53 s = 0.13%) is used in every panel and the caption says so.
PHOBERT_MEAN_TCC = 3360.89
T2_TCC = {"WikiBERT": (3077.08, 3.46492), "PhoBERT": (3360.89, 4.53214), "mBERT": (3181.59, 1.24008),
          "DistilBERT": (1554.86, 0.99322), "XLM-R": (3099.05, 1.41930), "CafeBERT": (9335.51, 148.46973),
          "viBERT": (3065.10, 1.53178), "vELECTRA": (3192.59, 1.57443), "ViSoBERT": (3211.82, 0.44646)}  # 3,201.82 in the submitted table was a typo

rep = []
def log(s=""):
    print(s); rep.append(str(s))

cms = {tuple([k.split("|")[0], int(k.split("|")[1])]): np.array(v) for k, v in json.load(open(ROOT / "results" / "confusion_matrices.json")).items()}


def scores(M):
    tp = np.diag(M); p = tp / M.sum(0); r = tp / M.sum(1); f = 2 * p * r / (p + r)
    return dict(acc=tp.sum() / M.sum(), P=p.mean(), R=r.mean(), F1=f.mean(),
                wF1=(f * M.sum(1)).sum() / M.sum(), f1_class=f)


tcc = {m: dict(JSON_TCC[m]) for m in JSON_TCC}
tcc["PhoBERT"] = {s: PHOBERT_MEAN_TCC for s in SEEDS}

log("Seed-specific TCC (s) and check against Table 2 (mean, population SD)")
for m in ORDER:
    v = [tcc[m][s] for s in SEEDS]
    log(f"  {m:10s} " + " ".join(f"{x:9.2f}" for x in v) + f" | mean {np.mean(v):8.2f} sd {np.std(v):7.2f}"
        f" | Table 2 {T2_TCC[m][0]:8.2f} sd {T2_TCC[m][1]:7.2f}")

# ------------------------------------------------------------ Fig 5 (four panels)
f1 = {(m, s): scores(cms[(m, s)])["F1"] * 100 for m in ORDER for s in SEEDS}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "font.weight": "bold"})
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4), sharex=True, sharey=True)
panels = [("(a) Seed 22", 22), ("(b) Seed 42", 42), ("(c) Seed 202", 202), ("(d) Mean of the three seeds", None)]
for ax, (title, s) in zip(axes.ravel(), panels):
    ax.set_facecolor("#EAEAF2"); ax.grid(color="white", lw=1); ax.set_axisbelow(True)
    xs, ys = [], []
    for m in ORDER:
        t = [tcc[m][q] for q in SEEDS]; f = [f1[(m, q)] for q in SEEDS]
        if s is None:
            x, y = np.log2(np.mean(t)), np.mean(f)
            sd_t = T2_TCC["PhoBERT"][1] if m == "PhoBERT" else np.std(t)
            ax.errorbar(x, y, xerr=[[x - np.log2(np.mean(t) - sd_t)], [np.log2(np.mean(t) + sd_t) - x]],
                        yerr=np.std(f), fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=2)
        else:
            x, y = np.log2(tcc[m][s]), f1[(m, s)]
        xs.append(x); ys.append(y)
        ax.scatter(x, y, marker=MARK[m], s=48, color=COLORS[m], edgecolor="black", lw=0.6, zorder=3,
                   label=LEG.get(m, m))
    ax.axhline(np.median(ys), ls="--", lw=0.8, color="grey"); ax.axvline(np.median(xs), ls="--", lw=0.8, color="grey")
    ax.set_title(title, fontweight="bold", fontsize=9)
for ax in axes[1]: ax.set_xlabel(r"$\log_2$ TCC (s)", fontweight="bold")
for ax in axes[:, 0]: ax.set_ylabel("Macro-F1 (%)", fontweight="bold")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=True)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "Fig5.pdf", bbox_inches="tight")
fig.savefig(OUT / "Fig5.tif", dpi=400, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})

# ------------------------------------------------------------ per-class F1 + weighted F1
rows = []
for m in ORDER:
    for s in SEEDS:
        sc = scores(cms[(m, s)])
        for c in range(10):
            rows.append(dict(model=m, seed=s, category=NAMES[c], f1=sc["f1_class"][c]))
        rows.append(dict(model=m, seed=s, category="__weighted_F1", f1=sc["wF1"]))
        rows.append(dict(model=m, seed=s, category="__macro_F1", f1=sc["F1"]))
pc = pd.DataFrame(rows); pc.to_csv(OUT / "per_class_f1.csv", index=False)
g = pc.groupby(["category", "model"]).f1.agg(["mean", lambda x: np.std(x)]).reset_index()
g.columns = ["category", "model", "mean", "sd"]
g["cell"] = (g["mean"] * 100).map("{:.2f}".format) + " (" + (g["sd"] * 100).map("{:.2f}".format) + ")"
wide = g.pivot(index="category", columns="model", values="cell")[ORDER]
wide = wide.loc[NAMES + ["__macro_F1", "__weighted_F1"]]
wide.to_csv(OUT / "per_class_f1_table.csv")
log("\nPer-category F1 (%), mean (population SD) over seeds"); log(wide.to_string())

# ------------------------------------------------------------ CafeBERT vs PhoBERT
# Article-level CafeBERT predictions are not available, so a paired test is impossible.
# Unpaired bootstrap: each run's test set is resampled independently from its own
# confusion matrix (multinomial over the 100 cells). Ignoring the positive correlation
# between two models evaluated on the same articles makes this interval WIDER than a
# paired interval, i.e. conservative.
rng = np.random.default_rng(20261106); B = 20000; N = 94670
def boot_f1(M):
    p = (M / M.sum()).ravel()
    draws = rng.multinomial(N, p, size=B).reshape(B, 10, 10)
    tp = np.einsum("bii->bi", draws); pr = tp / np.maximum(draws.sum(1), 1); rc = tp / np.maximum(draws.sum(2), 1)
    f = np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0)
    return f.mean(1)
res = []; pooled = []
for s in SEEDS:
    a, b = cms[("CafeBERT", s)], cms[("PhoBERT", s)]
    d_obs = (scores(a)["F1"] - scores(b)["F1"]) * 100
    d = (boot_f1(a) - boot_f1(b)) * 100; pooled.append(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    res.append(dict(seed=s, cafe_f1=scores(a)["F1"] * 100, phobert_f1=scores(b)["F1"] * 100, delta_pp=d_obs,
                    ci95_low=lo, ci95_high=hi, p_two_sided=min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                    cafe_correct=int(np.trace(a)), phobert_correct=int(np.trace(b))))
d = np.mean(pooled, axis=0); lo, hi = np.percentile(d, [2.5, 97.5])
res.append(dict(seed="mean", cafe_f1=np.mean([r["cafe_f1"] for r in res]), phobert_f1=np.mean([r["phobert_f1"] for r in res]),
                delta_pp=np.mean([r["delta_pp"] for r in res]), ci95_low=lo, ci95_high=hi,
                p_two_sided=min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
res = pd.DataFrame(res); res.to_csv(OUT / "cafe_vs_phobert.csv", index=False)
log(f"\nCafeBERT - PhoBERT macro-F1 (percentage points), unpaired bootstrap B={B}"); log(res.round(4).to_string(index=False))
(OUT / "report.txt").write_text("\n".join(rep), encoding="utf-8")
