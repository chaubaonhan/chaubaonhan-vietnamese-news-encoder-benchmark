"""STEP 7 - CafeBERT article-level analyses with the two retained seeds (42, 22). CPU, ~5 min.

Inputs (already on Drive from earlier notebooks):
  PLOS_revision/*/step4/G1_cafebert_seed{42,22}_pred.npz      (NB3, G1)
  PLOS_revision/*/revision_data/preds/PhoBERT_seed42.npz      (label order reference)
  PLOS_revision/*/revision_data/corpus_index.pkl              (publisher, year)
  PLOS_revision/*/revision_results/duplicates_test_flags.pkl  (NB2)
Outputs (PLOS_revision/NB7_CPU_cafebert_2seeds/):
  cafebert_per_source_year.csv, cafebert_duplicates_rescored.csv, step7_report.txt
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PLOS = Path(os.environ.get("PLOS_ROOT", "/content/drive/MyDrive/PLOS_revision"))
OUT = Path(os.environ.get("STEP7_OUT", PLOS / "NB7_CPU_cafebert_2seeds"))
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 22)
lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))
    (OUT / "step7_report.txt").write_text("\n".join(lines), encoding="utf-8")


def find_one(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        log(f"STOP: {what} not found ({pattern})")
        sys.exit(1)
    log(f"{what}: {hits[0]}")
    return hits[0]


f1 = lambda y, p: precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]
R = {s: dict(np.load(find_one(f"{PLOS}/*/step4/G1_cafebert_seed{s}_pred.npz", f"CafeBERT seed {s}")))
     for s in SEEDS}
y = np.load(find_one(f"{PLOS}/*/revision_data/preds/PhoBERT_seed42.npz", "PhoBERT seed 42"))["true"].astype(int)
for s in SEEDS:
    assert np.array_equal(R[s]["true"].astype(int), y), f"seed {s}: label order differs"
    log(f"CafeBERT seed {s}: n={len(y)}, correct={int(np.sum(R[s]['pred'] == y))}, "
        f"accuracy={np.mean(R[s]['pred'] == y):.6f}, macro_f1={f1(y, R[s]['pred']):.6f}")

ci = pd.read_pickle(find_one(f"{PLOS}/*/revision_data/corpus_index.pkl", "corpus_index.pkl"))
t = ci[ci.split == "test"].sort_values("pos")
assert np.array_equal(t["label"].to_numpy(), y), "corpus_index order differs from predictions"
rows = []
for col in ("source", "year"):
    vals = t[col].to_numpy()
    for v in sorted(pd.Series(vals).dropna().unique()):
        m = vals == v
        f1s = [f1(y[m], R[s]["pred"][m]) for s in SEEDS]
        accs = [np.mean(y[m] == R[s]["pred"][m]) for s in SEEDS]
        rows.append(dict(by=col, group=v, model="CafeBERT", n=int(m.sum()), macro_f1_mean=np.mean(f1s),
                         macro_f1_sd=np.std(f1s), acc_mean=np.mean(accs), acc_sd=np.std(accs)))
psy = pd.DataFrame(rows)
psy.to_csv(OUT / "cafebert_per_source_year.csv", index=False)
log("publisher / year (CafeBERT, seeds 42 and 22):\n" + psy.to_csv(index=False))

fl = pd.read_pickle(find_one(f"{PLOS}/*/revision_results/duplicates_test_flags.pkl", "duplicates_test_flags.pkl"))
dup = fl["any_duplicate"].to_numpy()
out = []
for name, m in [("full_test", np.ones_like(dup)), ("deduplicated_test", ~dup)]:
    f1s = [f1(y[m], R[s]["pred"][m]) for s in SEEDS]
    accs = [np.mean(y[m] == R[s]["pred"][m]) for s in SEEDS]
    out.append(dict(model="CafeBERT", subset=name, n=int(m.sum()), macro_f1_mean=np.mean(f1s),
                    macro_f1_sd=np.std(f1s), acc_mean=np.mean(accs), acc_sd=np.std(accs)))
out = pd.DataFrame(out)
out.to_csv(OUT / "cafebert_duplicates_rescored.csv", index=False)
log("de-duplicated test (CafeBERT, seeds 42 and 22):\n" + out.to_csv(index=False))
log("Done.")
