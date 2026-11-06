"""STEP 1b - Link the three private datasets row by row (CPU, high-RAM runtime).

  BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split  label(int), content  train/dev/test
  BaoNhan/Vietnamese_News_PhoBERT_Segmented        same rows, word-segmented content
  BaoNhan/Vietnam_News_3_Sources_Plus              url, label(str), content, source,
                                                   day, month, year (one split)

Output revision_data/corpus_index.pkl - one row per Cleaned article:
  split, pos, label (0-9 index used by the prediction files), source, year, url,
  plus_row (row in the Plus dataset, -1 if not found), h_exact, h_norm (content hashes)
and revision_data/step1b_report.txt. Every link is verified, nothing is assumed:
  * Cleaned and Segmented must have identical labels at every (split, pos)
  * the Cleaned test labels must equal the prediction-file labels (offset detected)
  * Plus rows are matched to Cleaned rows by an exact hash of the content
    (a whitespace-normalised hash is used only for rows without an exact match)
  * the category names implied by the match are checked against the manuscript
"""
import hashlib
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from common import DATA, LABEL_NAMES, MODEL_ORDER, SEEDS, pred_path

CLEAN = "BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split"
SEG = "BaoNhan/Vietnamese_News_PhoBERT_Segmented"
PLUS = "BaoNhan/Vietnam_News_3_Sources_Plus"
SPLIT_SIZES = {"train": 757356, "dev": 94670, "test": 94670}
report = []
WS = re.compile(r"\s+")


def log(msg=""):
    print(msg)
    report.append(str(msg))


def stop(msg):
    log("STOP: " + msg)
    (DATA / "step1b_report.txt").write_text("\n".join(report), encoding="utf-8")
    sys.exit(1)


def h_exact(t):
    return hashlib.blake2b((t or "").encode("utf-8"), digest_size=8).hexdigest()


def h_norm(t):
    return h_exact(WS.sub(" ", t or "").strip())


def read_repo(repo, prefix, columns):
    """Stream a split's parquet shards; content is reduced to hashes immediately."""
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(f"data/{prefix}-") and f.endswith(".parquet"))
    out = []
    for f in files:
        pf = pq.ParquetFile(hf_hub_download(repo, f, repo_type="dataset"))
        for b in pf.iter_batches(batch_size=20000, columns=columns):
            d = b.to_pandas()
            if "content" in d:
                d["h_exact"] = d["content"].map(h_exact)
                d["h_norm"] = d["content"].map(h_norm)
                d = d.drop(columns="content")
            out.append(d)
    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------- Cleaned and Segmented
parts = []
for split, n in SPLIT_SIZES.items():
    c = read_repo(CLEAN, split, ["label", "content"])
    s = read_repo(SEG, split, ["label"])
    if len(c) != n or len(s) != n:
        stop(f"{split}: Cleaned {len(c)} / Segmented {len(s)} rows, expected {n}")
    if not np.array_equal(c["label"].to_numpy(), s["label"].to_numpy()):
        stop(f"{split}: Cleaned and Segmented labels differ - rows are not aligned")
    c.insert(0, "split", split)
    c.insert(1, "pos", np.arange(n))
    parts.append(c)
    log(f"{split}: {n} rows, Cleaned/Segmented labels identical at every position; "
        f"label range {c['label'].min()}..{c['label'].max()}")
idx = pd.concat(parts, ignore_index=True)

# -------------------------------------------------------- label offset vs predictions
ref = next((pred_path(m, s) for m in MODEL_ORDER for s in SEEDS if pred_path(m, s).exists()), None)
if ref is None:
    stop("no prediction file found - run step1_collect.py first")
true = np.load(ref)["true"].astype(int)
test_lab = idx.loc[idx.split == "test", "label"].to_numpy()
offset = next((k for k in (0, 1) if np.array_equal(test_lab - k, true)), None)
if offset is None:
    stop("Cleaned test labels do not equal the prediction-file labels (offset 0 or 1)")
idx["label"] = idx["label"] - offset
log(f"Cleaned test split == prediction-file order; dataset label = index + {offset}")

# -------------------------------------------------------- Plus metadata
plus = read_repo(PLUS, "train", ["url", "label", "content", "source", "year"])
plus.insert(0, "plus_row", np.arange(len(plus)))
log(f"Plus: {len(plus)} rows; sources {plus.source.value_counts().to_dict()}")
for key in ("h_exact", "h_norm"):
    d = plus[key].duplicated(keep=False).sum()
    log(f"Plus rows sharing a {key} with another row: {d}")

m = idx.merge(plus.drop_duplicates("h_exact")[["h_exact", "plus_row"]], on="h_exact", how="left")
miss = m.plus_row.isna()
if miss.any():
    alt = plus.drop_duplicates("h_norm")[["h_norm", "plus_row"]].rename(columns={"plus_row": "pr2"})
    m = m.merge(alt, on="h_norm", how="left")
    m.loc[miss, "plus_row"] = m.loc[miss, "pr2"]
    m = m.drop(columns="pr2")
m["plus_row"] = m["plus_row"].fillna(-1).astype(int)
log(f"Cleaned rows matched to Plus: {(m.plus_row >= 0).mean():.4%} "
    f"(exact {(~miss).mean():.4%})")
meta = plus.set_index("plus_row")
ok = m.plus_row >= 0
for col in ("url", "source", "year"):
    m.loc[ok, col] = meta.loc[m.loc[ok, "plus_row"], col].to_numpy()
m.loc[ok, "plus_label"] = meta.loc[m.loc[ok, "plus_row"], "label"].to_numpy()

# -------------------------------------------------------- category-name check
ct = pd.crosstab(m.loc[ok, "label"], m.loc[ok, "plus_label"])
implied = ct.idxmax(axis=1).reindex(range(10)).tolist()
purity = (ct.max(axis=1) / ct.sum(axis=1)).round(4).tolist()
log(f"category name implied by Plus for index 0..9: {implied}")
log(f"share of rows agreeing with that name: {purity}")
log(f"manuscript names for index 0..9:          {LABEL_NAMES}")
log("NAMES AGREE" if implied == LABEL_NAMES else "NAMES DIFFER - report this before using them")
for split in SPLIT_SIZES:
    s = m[m.split == split]
    log(f"{split}: years {s.year.value_counts().sort_index().to_dict()}")

m.to_pickle(DATA / "corpus_index.pkl")
(DATA / "step1b_report.txt").write_text("\n".join(report), encoding="utf-8")
print("Done.")
