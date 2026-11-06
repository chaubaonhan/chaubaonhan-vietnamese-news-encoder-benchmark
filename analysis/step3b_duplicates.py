"""STEP 3b - Cross-partition duplicate audit (Reviewer #2, Concern 2). CPU, high-RAM.

Exact duplicates : identical content hash (raw, and whitespace-normalised).
Near duplicates  : MinHash-LSH on lower-cased word 5-gram shingles,
                   estimated Jaccard similarity >= THRESHOLD (default 0.8).
For every dev/test article we record whether it has an exact or near duplicate
in train (and test vs dev). The released predictions are then re-scored on the
test articles with no duplicate in train or dev.

Self-contained MinHash-LSH (numpy only).
Outputs (revision_results/): duplicates_summary.csv, duplicates_test_flags.pkl,
duplicates_rescored.csv, step3b_report.txt
"""
import os
import re
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import precision_recall_fscore_support

from common import DATA, MODEL_ORDER, OUT, SEEDS, pred_path

CLEAN = "BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split"
THRESHOLD = float(os.environ.get("NEARDUP_THRESHOLD", 0.8))
NUM_PERM, BANDS = 128, 16          # 16 bands x 8 rows: candidate recall high at J >= 0.8
ROWS = NUM_PERM // BANDS
SHINGLE = 5
MAX_CAND = 500                      # candidates checked per query and band
OUT.mkdir(parents=True, exist_ok=True)
report = []
WS = re.compile(r"\s+")
_P = np.uint64((1 << 31) - 1)
_rng = np.random.default_rng(1)
_A = _rng.integers(1, (1 << 31) - 1, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, (1 << 31) - 1, NUM_PERM, dtype=np.uint64)


def log(msg=""):
    print(msg, flush=True)
    report.append(str(msg))


def signature(text):
    """Self-contained MinHash (no external library): deterministic across processes."""
    words = WS.sub(" ", (text or "").lower()).strip().split(" ")
    sh = {" ".join(words[i:i + SHINGLE]) for i in range(max(1, len(words) - SHINGLE + 1))}
    x = np.fromiter((zlib.crc32(t.encode("utf-8")) for t in sh), dtype=np.uint64, count=len(sh))
    return ((np.outer(x, _A) + _B) % _P).min(axis=0).astype(np.uint32)


def signatures(split):
    files = sorted(f for f in HfApi().list_repo_files(CLEAN, repo_type="dataset")
                   if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    sigs = []
    with Pool(os.cpu_count()) as pool:
        for f in files:
            pf = pq.ParquetFile(hf_hub_download(CLEAN, f, repo_type="dataset"))
            for b in pf.iter_batches(batch_size=20000, columns=["content"]):
                sigs += pool.map(signature, b.column("content").to_pylist(), chunksize=500)
            log(f"  {split}: {len(sigs)} signatures")
    return np.vstack(sigs)


def band_keys(sig):
    mult = np.uint64(0x9E3779B97F4A7C15)
    keys = np.zeros((len(sig), BANDS), dtype=np.uint64)
    s = sig.astype(np.uint64)
    with np.errstate(over="ignore"):
        for b in range(BANDS):
            k = np.zeros(len(sig), dtype=np.uint64)
            for r in range(ROWS):
                k = k * mult + s[:, b * ROWS + r]
            keys[:, b] = k
    return keys


def near_flags(q_sig, r_sig):
    """True for each query with an estimated Jaccard >= THRESHOLD to any reference."""
    qk, rk = band_keys(q_sig), band_keys(r_sig)
    flag = np.zeros(len(q_sig), dtype=bool)
    for b in range(BANDS):
        order = np.argsort(rk[:, b], kind="stable")
        sorted_keys = rk[order, b]
        lo = np.searchsorted(sorted_keys, qk[:, b], "left")
        hi = np.searchsorted(sorted_keys, qk[:, b], "right")
        for i in np.where((hi > lo) & ~flag)[0]:
            cand = order[lo[i]:min(hi[i], lo[i] + MAX_CAND)]
            if (np.mean(r_sig[cand] == q_sig[i], axis=1) >= THRESHOLD).any():
                flag[i] = True
    return flag


idx = pd.read_pickle(DATA / "corpus_index.pkl")

# ------------------------------------------------------------------ exact
flags = {}
for key in ("h_exact", "h_norm"):
    tr = set(idx.loc[idx.split == "train", key])
    dv = set(idx.loc[idx.split == "dev", key])
    for split in ("dev", "test"):
        s = idx[idx.split == split]
        flags[(split, key, "train")] = s[key].isin(tr).to_numpy()
        if split == "test":
            flags[(split, key, "dev")] = s[key].isin(dv).to_numpy()

# ------------------------------------------------------------------ near
log(f"MinHash: {NUM_PERM} permutations, {BANDS} bands x {ROWS} rows, "
    f"{SHINGLE}-word shingles, threshold={THRESHOLD}")
sig = {s: signatures(s) for s in ("train", "dev", "test")}
flags[("dev", "near", "train")] = near_flags(sig["dev"], sig["train"])
flags[("test", "near", "train")] = near_flags(sig["test"], sig["train"])
flags[("test", "near", "dev")] = near_flags(sig["test"], sig["dev"])

rows = [dict(query_split=q, kind=k, reference_split=r, n_flagged=int(v.sum()),
             pct=100 * v.mean()) for (q, k, r), v in flags.items()]
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "duplicates_summary.csv", index=False)
log(summ.round(3).to_string(index=False))

test_dup = np.zeros(int((idx.split == "test").sum()), dtype=bool)
for (q, k, r), v in flags.items():
    if q == "test":
        test_dup |= v
pd.DataFrame({k[1] + "_in_" + k[2]: v for k, v in flags.items() if k[0] == "test"}
             ).assign(any_duplicate=test_dup).to_pickle(OUT / "duplicates_test_flags.pkl")
log(f"test articles with any exact or near duplicate in train/dev: "
    f"{test_dup.sum()} ({100*test_dup.mean():.2f}%)")

# ------------------------------------------------------------------ rescore
res = []
for m in MODEL_ORDER:
    if not all(pred_path(m, s).exists() for s in SEEDS):
        log(f"{m}: predictions missing, not rescored")
        continue
    for name, mask in [("full_test", np.ones_like(test_dup)), ("deduplicated_test", ~test_dup)]:
        f1, acc = [], []
        for s in SEEDS:
            r = np.load(pred_path(m, s))
            y, p = r["true"][mask], r["pred"][mask]
            f1.append(precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2])
            acc.append(np.mean(y == p))
        res.append(dict(model=m, subset=name, n=int(mask.sum()), macro_f1_mean=np.mean(f1),
                        macro_f1_sd=np.std(f1), acc_mean=np.mean(acc), acc_sd=np.std(acc)))
res = pd.DataFrame(res)
res.to_csv(OUT / "duplicates_rescored.csv", index=False)
log(res.pivot(index="model", columns="subset", values="macro_f1_mean").mul(100).round(3).to_string())
(OUT / "step3b_report.txt").write_text("\n".join(report), encoding="utf-8")
print("Done.")
