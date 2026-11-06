"""STEP 3 - Model-specific truncation audit on the test split (R1 C7, R2 C3).

Runs on CPU (a GPU is not needed). Uses the tokenizer saved with each released
checkpoint (seed 42).

Outputs in revision_results/:
  truncation_stats.csv / .tex     per checkpoint: token-length distribution,
                                  number and % of truncated articles, retained share
  truncation_performance.csv      macro-F1 / accuracy on truncated vs. untruncated
                                  articles (mean, population SD over seeds)
  token_lengths.npz               per-article token counts (reused by later steps)
  step3_report.txt

Inputs are the test splits of BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split
(eight models) and BaoNhan/Vietnamese_News_PhoBERT_Segmented (PhoBERT).
"""
import os
import sys

import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer

from common import (BASE_CHECKPOINTS, DATA, MAX_LEN, MODEL_ORDER, MODELS, OUT, SEEDS,
                    pred_path)

OUT.mkdir(parents=True, exist_ok=True)
report = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


def tokenizer_location(model):
    repo, key = MODELS[model]
    if repo is not None:
        return repo, f"{key}_checkpoints_seed_42/checkpoint-8877"
    return BASE_CHECKPOINTS[model], None  # CafeBERT: tokenizer of the public checkpoint


# ------------------------------------------------------------ model inputs
# Same test texts that were used in training: the Cleaned split for eight models,
# the word-segmented split for PhoBERT (datasets named by the authors).
CLEAN = "BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split"
SEG = "BaoNhan/Vietnamese_News_PhoBERT_Segmented"


def load_test(repo):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith("data/test-") and f.endswith(".parquet"))
    tab = pd.concat([pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pandas()
                     for f in files], ignore_index=True)
    true = next(np.load(pred_path(m, s))["true"] for m in MODEL_ORDER for s in SEEDS
                if pred_path(m, s).exists()).astype(int)
    lab = tab["label"].to_numpy()
    if not any(np.array_equal(lab - k, true) for k in (0, 1)):
        sys.exit(f"{repo}: test labels are not in prediction-file order")
    return tab["content"].fillna("").astype(str).tolist()


texts_clean = load_test(CLEAN)
texts_seg = load_test(SEG)
log(f"test articles: {len(texts_clean)} (Cleaned) / {len(texts_seg)} (PhoBERT-segmented)")

cache = OUT / "token_lengths.npz"
lengths = dict(np.load(cache)) if cache.exists() else {}
n_special = {}
for m in MODEL_ORDER:
    repo, sub = tokenizer_location(m)
    tok = (AutoTokenizer.from_pretrained(repo, subfolder=sub) if sub
           else AutoTokenizer.from_pretrained(repo))
    n_special[m] = tok.num_special_tokens_to_add(pair=False)
    if m in lengths:
        continue
    inp = texts_seg if m == "PhoBERT" else texts_clean
    lens = []
    for i in range(0, len(inp), 2000):
        enc = tok(inp[i:i + 2000], add_special_tokens=True, truncation=False,
                  return_attention_mask=False, verbose=False)["input_ids"]
        lens += [len(x) for x in enc]
    lengths[m] = np.array(lens)
    np.savez_compressed(cache, **lengths)
    log(f"{m}: tokenizer {repo}/{sub} ({type(tok).__name__}), "
        f"{n_special[m]} special tokens")

# ------------------------------------------------------------ statistics
rows = []
for m in MODEL_ORDER:
    L = lengths[m]
    trunc = L > MAX_LEN
    kept = np.minimum(1.0, (MAX_LEN - n_special[m]) / np.maximum(L - n_special[m], 1))
    q1, q3 = np.percentile(kept[trunc], [25, 75]) if trunc.any() else (1.0, 1.0)
    rows.append(dict(
        model=m, mean_tokens=L.mean(), median_tokens=np.median(L),
        p90_tokens=np.percentile(L, 90), n_truncated=int(trunc.sum()),
        pct_truncated=100 * trunc.mean(),
        mean_retained_pct_all=100 * kept.mean(),
        median_retained_pct_truncated=100 * np.median(kept[trunc]) if trunc.any() else 100,
        iqr_retained_pct_truncated=f"{100*q1:.1f}-{100*q3:.1f}"))
st = pd.DataFrame(rows)
st.to_csv(OUT / "truncation_stats.csv", index=False)
(OUT / "truncation_stats.tex").write_text(
    st.to_latex(index=False, float_format="%.1f"), encoding="utf-8")
log("\nTruncation statistics (test split, budget = %d tokens incl. special tokens):" % MAX_LEN)
log(st.round(1).to_string(index=False))

# ------------------------------------------------------------ performance
perf = []
masks = {m: lengths[m] > MAX_LEN for m in MODEL_ORDER}
all_trunc = np.logical_and.reduce(list(masks.values()))
none_trunc = ~np.logical_or.reduce(list(masks.values()))
log(f"\narticles truncated for all nine checkpoints: {all_trunc.sum()}; "
    f"for none: {none_trunc.sum()}")
for m in MODEL_ORDER:
    if not all(pred_path(m, s).exists() for s in SEEDS):
        log(f"{m}: predictions missing, performance split skipped")
        continue
    for name, mask in [("own_truncated", masks[m]), ("own_untruncated", ~masks[m]),
                       ("truncated_for_all", all_trunc), ("truncated_for_none", none_trunc)]:
        f1s, accs = [], []
        for s in SEEDS:
            r = np.load(pred_path(m, s))
            y, yhat = r["true"][mask], r["pred"][mask]
            f1s.append(precision_recall_fscore_support(y, yhat, average="macro",
                                                       zero_division=0)[2])
            accs.append(np.mean(y == yhat))
        perf.append(dict(model=m, subset=name, n=int(mask.sum()),
                         macro_f1_mean=np.mean(f1s), macro_f1_sd=np.std(f1s),
                         acc_mean=np.mean(accs), acc_sd=np.std(accs)))
perf = pd.DataFrame(perf)
perf.to_csv(OUT / "truncation_performance.csv", index=False)
log(perf.pivot(index="model", columns="subset", values="macro_f1_mean")
    .mul(100).round(2).to_string())
(OUT / "step3_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nDone.")
