"""STEP 2 - Statistics that need no GPU (Reviewer #1 C4, C5, C8; Reviewer #2 C1).

Reads revision_data/ (from step1) and writes revision_results/:
  step2_report.txt            all checks and headline numbers
  per_seed_metrics.csv        recomputed metrics for every model-seed run
  tcc_per_seed.csv            seed-specific TCC = train_duration + predict_duration
  Fig5.pdf / Fig5.tif         four-panel performance-cost figure (R1 C4)
  cafe_vs_phobert.csv         paired bootstrap + McNemar per seed and pooled (R1 C5)
  pairwise_bootstrap.csv      seed-averaged delta macro-F1 for all 36 pairs, Holm-adjusted
  per_class_f1.csv / .tex     per-category F1, mean (SD) over seeds (R1 C8)
  per_source_year.csv         macro-F1/accuracy per publisher and per year (R2 C1),
                              using revision_data/corpus_index.pkl from step1b

Optional input: manual_tcc.csv (columns model,seed,tcc_seconds) for runs whose
released JSON has no train_duration/predict_duration.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from common import (COLORS, DATA, FIG_SEED_ORDER, LABEL_NAMES, MODEL_ORDER,
                    N_CLASSES, OUT, SEEDS, TABLE2, TABLE2_TCC_SD, pred_path)

import os
B = int(os.environ.get("BOOT_B", 10_000))
RNG = np.random.default_rng(20261106)
OUT.mkdir(parents=True, exist_ok=True)
report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


def save_report():
    (OUT / "step2_report.txt").write_text("\n".join(report), encoding="utf-8")


# ------------------------------------------------------------------ load
runs = {}
for m in MODEL_ORDER:
    for s in SEEDS:
        p = pred_path(m, s)
        if p.exists():
            runs[(m, s)] = dict(np.load(p))
models = [m for m in MODEL_ORDER if all((m, s) in runs for s in SEEDS)]
log(f"models with all three seeds: {models}")
log(f"models skipped (missing runs): {[m for m in MODEL_ORDER if m not in models]}")
if not models:
    save_report()
    sys.exit("no complete model found - run step1 first")
TRUE = runs[(models[0], SEEDS[0])]["true"].astype(np.int64)
N = len(TRUE)


def macro_f1_from_conf(conf):
    conf = conf.reshape(N_CLASSES, N_CLASSES)  # rows = true, cols = pred
    tp = np.diag(conf)
    prec = np.divide(tp, conf.sum(0), out=np.zeros(N_CLASSES), where=conf.sum(0) > 0)
    rec = np.divide(tp, conf.sum(1), out=np.zeros(N_CLASSES), where=conf.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(N_CLASSES),
                   where=(prec + rec) > 0)
    return f1.mean()


# ------------------------------------------------------------------ 1. metrics
rows = []
for (m, s), r in runs.items():
    y, yhat = r["true"], r["pred"]
    p, rc, f, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y, yhat, average="weighted", zero_division=0)
    auc = roc_auc_score(y, r["prob"], multi_class="ovr", average="macro")
    rows.append(dict(model=m, seed=s, accuracy=np.mean(y == yhat), macro_precision=p,
                     macro_recall=rc, macro_f1=f, weighted_f1=fw, macro_auc=auc))
met = pd.DataFrame(rows)
met.to_csv(OUT / "per_seed_metrics.csv", index=False)
log("\n[1] Recomputed means vs Table 2 (difference in the 5th decimal is rounding):")
for m in models:
    g = met[met.model == m]
    t = TABLE2[m]
    log(f"  {m:10s} acc {g.accuracy.mean():.5f} (T2 {t[1]:.5f})  "
        f"F1 {g.macro_f1.mean():.5f} (T2 {t[4]:.5f})  AUC {g.macro_auc.mean():.5f} "
        f"(T2 {t[5]:.5f})  weighted-F1 {g.weighted_f1.mean():.5f}")

# ------------------------------------------------------------------ 2. TCC
def find_seed_block(obj, seed):
    """Return the dict holding seed-specific metrics, searching nested JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in (f"seed_{seed}", str(seed), f"seed{seed}") and isinstance(v, dict):
                return v
        for v in obj.values():
            hit = find_seed_block(v, seed)
            if hit is not None:
                return hit
    return None


tcc = {}
for m in models:
    f = DATA / "metrics" / f"{m}.json"
    if not f.exists():
        continue
    merged = json.load(open(f, encoding="utf-8"))
    for s in SEEDS:
        blk = None
        for fname, content in merged.items():
            if m == "CafeBERT" and f"seed{s}" not in fname:
                continue  # one repository per seed
            blk = find_seed_block(content, s)
            if blk is None and m == "CafeBERT" and isinstance(content, dict):
                blk = content
            if blk and "train_duration" in blk and "predict_duration" in blk:
                tcc[(m, s)] = blk["train_duration"] + blk["predict_duration"]
                break
manual = Path("manual_tcc.csv")
if manual.exists():
    for _, r in pd.read_csv(manual).iterrows():
        tcc.setdefault((r["model"], int(r["seed"])), float(r["tcc_seconds"]))
        log(f"  manual TCC used for {r['model']} seed {int(r['seed'])}")
tdf = pd.DataFrame([dict(model=m, seed=s, tcc_seconds=v) for (m, s), v in tcc.items()])
tdf.to_csv(OUT / "tcc_per_seed.csv", index=False)
log("\n[2] Seed-specific TCC (s) vs Table 2 (mean and population SD):")
tcc_models = []
for m in models:
    v = [tcc.get((m, s)) for s in SEEDS]
    if None in v:
        log(f"  {m:10s} MISSING for seeds {[s for s, x in zip(SEEDS, v) if x is None]}")
        continue
    tcc_models.append(m)
    log(f"  {m:10s} " + "  ".join(f"s{s}={x:9.2f}" for s, x in zip(SEEDS, v)) +
        f"  mean={np.mean(v):9.2f} (T2 {TABLE2[m][6]:9.2f})  "
        f"sd={np.std(v):8.3f} (T2 {TABLE2_TCC_SD[m]:8.3f})")

# ------------------------------------------------------------------ 3. Fig 5
MARK = {"WikiBERT": "o", "PhoBERT": "s", "mBERT": "^", "DistilBERT": "D",
        "XLM-R": "P", "CafeBERT": "X", "viBERT": "v", "vELECTRA": "<", "ViSoBERT": ">"}
LEG = {"PhoBERT": "PhoBERT-base", "XLM-R": "XLM-R-base", "vELECTRA": "vELECTRA-base"}
missing_fig = [m for m in MODEL_ORDER if m not in tcc_models]
if missing_fig:
    log(f"\n[3] Fig 5 NOT drawn: no seed-specific TCC for {missing_fig}")
else:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6), sharex=True, sharey=True)
    panels = [(f"({c}) Seed {s}", s) for c, s in zip("abc", FIG_SEED_ORDER)]
    panels.append(("(d) Mean over the three seeds", None))
    for ax, (title, s) in zip(axes.ravel(), panels):
        ax.set_facecolor("#EAEAF2")
        ax.grid(color="white", linewidth=1)
        ax.set_axisbelow(True)
        xs, ys = [], []
        for m in MODEL_ORDER:
            f1s = [met[(met.model == m) & (met.seed == q)].macro_f1.item() * 100 for q in SEEDS]
            ts = [tcc[(m, q)] for q in SEEDS]
            if s is None:
                x, y = np.log2(np.mean(ts)), np.mean(f1s)
                lo = x - np.log2(np.mean(ts) - np.std(ts))
                hi = np.log2(np.mean(ts) + np.std(ts)) - x
                ax.errorbar(x, y, xerr=[[lo], [hi]], yerr=np.std(f1s), fmt="none",
                            ecolor="black", elinewidth=0.8, capsize=2, zorder=2)
            else:
                x, y = np.log2(tcc[(m, s)]), f1s[SEEDS.index(s)]
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y, marker=MARK[m], s=46, color=COLORS[m], edgecolor="black",
                       linewidth=0.6, zorder=3, label=LEG.get(m, m))
        ax.axhline(np.median(ys), ls="--", lw=0.8, color="grey")
        ax.axvline(np.median(xs), ls="--", lw=0.8, color="grey")
        ax.set_title(title, fontweight="bold", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel(r"$\log_2$ TCC (s)", fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (%)", fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "Fig5.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig5.tif", dpi=400, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    log("\n[3] Fig 5 saved (panels a-c: seed-specific TCC and macro-F1; "
        "panel d: mean TCC vs mean macro-F1, error bars = population SD; "
        "dashed lines = medians across the nine checkpoints)")

# ------------------------------------------------------------------ 4. tests
log("\n[4] Paired tests")
codes = {k: TRUE * N_CLASSES + v["pred"].astype(np.int64) for k, v in runs.items()
         if k[0] in models}
obs_f1 = {k: macro_f1_from_conf(np.bincount(c, minlength=100)) for k, c in codes.items()}
pairs = list(combinations(models, 2))
boot = {k: np.empty(B) for k in codes}
for b in range(B):
    w = np.bincount(RNG.integers(0, N, N), minlength=N)
    for k, c in codes.items():
        boot[k][b] = macro_f1_from_conf(np.bincount(c, weights=w, minlength=100))


def summarize(delta_obs, delta_boot):
    lo, hi = np.percentile(delta_boot, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(delta_boot <= 0), np.mean(delta_boot >= 0)))
    return delta_obs, lo, hi, p


rows = []
if "CafeBERT" in models and "PhoBERT" in models:
    for s in SEEDS:
        a, bm = runs[("CafeBERT", s)], runs[("PhoBERT", s)]
        ca, cb = a["pred"] == a["true"], bm["pred"] == bm["true"]
        n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        d, lo, hi, p = summarize(obs_f1[("CafeBERT", s)] - obs_f1[("PhoBERT", s)],
                                 boot[("CafeBERT", s)] - boot[("PhoBERT", s)])
        chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        rows.append(dict(seed=s, delta_macro_f1_pp=d * 100, ci_low_pp=lo * 100,
                         ci_high_pp=hi * 100, bootstrap_p=p,
                         delta_accuracy_pp=(ca.mean() - cb.mean()) * 100,
                         cafe_only_correct=n01, phobert_only_correct=n10,
                         net_articles=n01 - n10,
                         mcnemar_exact_p=binomtest(n01, n01 + n10, 0.5).pvalue,
                         mcnemar_chi2_cc=chi, mcnemar_chi2_p=chi2.sf(chi, 1)))
    d_obs = np.mean([obs_f1[("CafeBERT", s)] - obs_f1[("PhoBERT", s)] for s in SEEDS])
    d_boot = np.mean([boot[("CafeBERT", s)] - boot[("PhoBERT", s)] for s in SEEDS], axis=0)
    d, lo, hi, p = summarize(d_obs, d_boot)
    rows.append(dict(seed="mean", delta_macro_f1_pp=d * 100, ci_low_pp=lo * 100,
                     ci_high_pp=hi * 100, bootstrap_p=p))
    cvp = pd.DataFrame(rows)
    cvp.to_csv(OUT / "cafe_vs_phobert.csv", index=False)
    log("  CafeBERT - PhoBERT (percentage points), B = %d:" % B)
    log(cvp.to_string(index=False))
else:
    log("  CafeBERT vs PhoBERT skipped: one of them has no complete prediction set")

pw = []
for a, bm in pairs:
    d_obs = np.mean([obs_f1[(a, s)] - obs_f1[(bm, s)] for s in SEEDS])
    d_boot = np.mean([boot[(a, s)] - boot[(bm, s)] for s in SEEDS], axis=0)
    d, lo, hi, p = summarize(d_obs, d_boot)
    pw.append(dict(model_a=a, model_b=bm, delta_macro_f1_pp=d * 100,
                   ci_low_pp=lo * 100, ci_high_pp=hi * 100, p=p))
pw = pd.DataFrame(pw).sort_values("p").reset_index(drop=True)
k = len(pw)
pw["p_holm"] = np.minimum(1, np.maximum.accumulate(pw.p * (k - np.arange(k))))
pw.to_csv(OUT / "pairwise_bootstrap.csv", index=False)
log(f"  all {k} pairs written to pairwise_bootstrap.csv; "
    f"{int((pw.p_holm < 0.05).sum())} remain significant after Holm (alpha 0.05)")

# ------------------------------------------------------------------ 5. per class
pc = []
for (m, s), r in runs.items():
    if m not in models:
        continue
    _, _, f, _ = precision_recall_fscore_support(r["true"], r["pred"], labels=range(10),
                                                 average=None, zero_division=0)
    for c in range(N_CLASSES):
        pc.append(dict(model=m, seed=s, category=LABEL_NAMES[c], f1=f[c]))
pc = pd.DataFrame(pc)
pc.to_csv(OUT / "per_class_f1.csv", index=False)
tab = pc.groupby(["category", "model"]).f1.agg(["mean", lambda x: np.std(x)]).reset_index()
tab.columns = ["category", "model", "mean", "sd"]
cells = tab.assign(cell=lambda d: d["mean"].map(lambda v: f"{v*100:.2f}") + " (" +
                   d["sd"].map(lambda v: f"{v*100:.2f}") + ")")
wide = cells.pivot(index="category", columns="model", values="cell")[models]
wide = wide.loc[[c for c in LABEL_NAMES]]
(OUT / "per_class_f1.tex").write_text(wide.to_latex(escape=True), encoding="utf-8")
log("\n[5] Per-category F1 (%), mean (SD) over seeds:")
log(wide.to_string())

# ------------------------------------------------------------------ 6. sources
log("\n[6] Breakdown by publisher and by publication year (original test split)")
try:
    ci = pd.read_pickle(DATA / "corpus_index.pkl")
    t = ci[ci.split == "test"].sort_values("pos")
    if not np.array_equal(t["label"].to_numpy(), TRUE):
        raise ValueError("corpus_index test labels are not in prediction order - rerun step1b")
    log(f"  test articles with publisher/year metadata: {(t.plus_row >= 0).mean():.4%}")
    out = []
    for col in ("source", "year"):
        vals = t[col].to_numpy()
        for v in sorted(pd.Series(vals).dropna().unique()):
            mask = vals == v
            for m in models:
                f1s = [precision_recall_fscore_support(runs[(m, s)]["true"][mask], runs[(m, s)]["pred"][mask],
                                                       average="macro", zero_division=0)[2] for s in SEEDS]
                accs = [np.mean(runs[(m, s)]["true"][mask] == runs[(m, s)]["pred"][mask]) for s in SEEDS]
                out.append(dict(by=col, group=v, model=m, n=int(mask.sum()), macro_f1_mean=np.mean(f1s),
                                macro_f1_sd=np.std(f1s), acc_mean=np.mean(accs), acc_sd=np.std(accs)))
    out = pd.DataFrame(out)
    out.to_csv(OUT / "per_source_year.csv", index=False)
    for col in ("source", "year"):
        log(out[out.by == col].pivot(index="model", columns="group", values="macro_f1_mean")
            .loc[models].mul(100).round(2).to_string())
except Exception as e:  # noqa: BLE001
    log(f"  skipped: {e}")

save_report()
print("\nDone. Send the whole revision_results/ folder.")
