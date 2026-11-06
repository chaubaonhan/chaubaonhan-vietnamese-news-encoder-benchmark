"""STEP 1 - Download the released predictions/metrics and assemble them.

Runs on CPU. Requires `huggingface_hub`, `datasets`, `pandas`, `numpy`.
Output folder: revision_data/
  preds/<Model>_seed<s>.npz   true, pred, prob (aligned row by row)
  test_text.pkl           row, id, content of the 94,670 test articles
  metrics/<Model>.json        raw per-seed metric JSON(s) as released
  dataset_meta.pkl        label/url/source of BaoNhan/Vietnam_News_3_Sources
  step1_report.txt            every check performed and its outcome

The script STOPS with an explanation if any consistency check fails.
"""
import json
import sys

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from common import (DATA, MODELS, N_TEST, SEEDS, TEST_COUNTS,
                    pred_path)

api = HfApi()
(DATA / "preds").mkdir(parents=True, exist_ok=True)
(DATA / "metrics").mkdir(parents=True, exist_ok=True)
report = []


def log(msg):
    print(msg)
    report.append(msg)


def fail(msg):
    log("STOP: " + msg)
    (DATA / "step1_report.txt").write_text("\n".join(report), encoding="utf-8")
    sys.exit(1)


# ---------------------------------------------------------------- locate files
locations = {}  # (model, seed) -> (repo, csv path)
metric_files = {}  # model -> list of (repo, path)
import os
ONLY = [m.strip() for m in os.environ.get("ONLY_MODELS", "").split(",") if m.strip()]
for model, (repo, key) in MODELS.items():
    if repo is None or (ONLY and model not in ONLY):
        continue
    files = api.list_repo_files(repo)
    for s in SEEDS:
        p = f"results_{key}/predictions_seed_{s}.csv"
        if p not in files:
            fail(f"{repo} has no {p}")
        locations[(model, s)] = (repo, p)
    metric_files[model] = [(repo, f) for f in files
                           if f.startswith(f"results_{key}/") and f.endswith(".json")]

metric_files["CafeBERT"] = []
log("CafeBERT: fine-tuned predictions are not released on the Hub - skipped")

# ---------------------------------------------------------------- predictions
PROB_COLS = [f"prob_class_{k}" for k in range(10)]
ref_true, ref_id, text_saved = None, None, False
for (model, s), (repo, path) in sorted(locations.items()):
    local = hf_hub_download(repo, path)
    header = pd.read_csv(local, nrows=0).columns.tolist()
    use = [c for c in header if c != "content"]
    df = pd.read_csv(local, usecols=use)
    if len(df) != N_TEST:
        fail(f"{model} seed {s}: {len(df)} rows, expected {N_TEST}")
    for c in ["true_label_0_9", "predicted_label_0_9"] + PROB_COLS:
        if c not in df.columns:
            fail(f"{model} seed {s}: column {c} missing; columns = {header}")
    true = df["true_label_0_9"].to_numpy(np.int8)
    pred = df["predicted_label_0_9"].to_numpy(np.int8)
    prob = df[PROB_COLS].to_numpy(np.float32)
    if ref_true is None:
        ref_true = true
        counts = np.bincount(true, minlength=10).tolist()
        if counts != TEST_COUNTS:
            fail(f"class counts {counts} differ from manuscript {TEST_COUNTS}")
        log(f"class counts match the manuscript: {counts}")
    elif not np.array_equal(true, ref_true):
        fail(f"{model} seed {s}: true-label order differs from the reference file")
    if "id" in df.columns:
        ids = df["id"].to_numpy()
        if ref_id is None:
            ref_id = ids
        elif not np.array_equal(ids, ref_id):
            fail(f"{model} seed {s}: id order differs from the reference file")
        if not text_saved:
            txt = pd.read_csv(local, usecols=["id", "content"])
            txt.insert(0, "row", np.arange(len(txt)))
            txt.to_pickle(DATA / "test_text.pkl")
            text_saved = True
            log(f"test text saved from {repo}/{path}")
    np.savez_compressed(pred_path(model, s), true=true, pred=pred, prob=prob)
    log(f"{model:10s} seed {s:3d}: OK  acc={np.mean(true == pred):.5f}  "
        f"id column={'id' in df.columns}")

missing = [(m, s) for m in MODELS for s in SEEDS if (m, s) not in locations and (not ONLY or m in ONLY)]
log(f"prediction files missing: {missing if missing else 'none'}")

# ---------------------------------------------------------------- metrics json
for model, lst in metric_files.items():
    merged = {}
    for repo, f in lst:
        merged[f"{repo}/{f}"] = json.load(open(hf_hub_download(repo, f), encoding="utf-8"))
    (DATA / "metrics" / f"{model}.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{model}: {len(lst)} metric JSON file(s) saved")

# ---------------------------------------------------------------- dataset meta
try:
    from datasets import load_dataset
    ds = load_dataset("BaoNhan/Vietnam_News_3_Sources")["train"].to_pandas()
    ds.insert(0, "row", np.arange(len(ds)))
    ds.to_pickle(DATA / "dataset_meta.pkl")
    log(f"dataset metadata saved: {len(ds)} rows, columns {list(ds.columns)}")
    if ref_id is not None:
        log(f"test id examples: {ref_id[:5].tolist()}  dtype={ref_id.dtype}")
except Exception as e:  # noqa: BLE001
    log(f"dataset metadata NOT saved: {e!r}")

(DATA / "step1_report.txt").write_text("\n".join(report), encoding="utf-8")
print("\nDone. Send revision_data/step1_report.txt.")
