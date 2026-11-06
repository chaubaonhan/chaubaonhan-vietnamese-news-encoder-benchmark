"""STEP 4 - The only GPU work for the revision (Colab G4, about 3-3.5 GPU-hours).

  G1  CafeBERT inference (seeds 42 and 22, checkpoints in Drive/CafeBERT_Results)
      -> article-level predictions -> PAIRED McNemar test and paired bootstrap of
         macro-F1 against PhoBERT-base (Reviewer #1, Concern 5).            ~10 min
  G2  PhoBERT-base, seed 42: head-only truncation (baseline, rerun on G4) versus
      head+tail truncation (first + last tokens within the same 256 budget)
      (Reviewer #2, Concern 3).                                              ~2 h
  G3  PhoBERT-base, seed 42: temporal hold-out - train/dev on 2019-2022 articles,
      test separately on all 2023, 2024 and 2025 articles (Reviewer #2, Concern 1).
                                                                             ~45 min
Every run is saved to Google Drive and skipped when re-launched (resumable).
Requires the outputs of step1_collect.py and step1b_index.py in revision_data/.

    from google.colab import drive; drive.mount('/content/drive')
    !python step4_gpu.py            # all stages
    !python step4_gpu.py G1         # one stage
"""
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from common import DATA, SEEDS, pred_path

# ----------------------------------------------------------------- configuration
OUT = Path(os.environ.get("STEP4_OUT", "/content/drive/MyDrive/PLOS_revision_step4"))
OUT.mkdir(parents=True, exist_ok=True)
SEG = "BaoNhan/Vietnamese_News_PhoBERT_Segmented"
CLEAN = "BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split"
PHOBERT = "vinai/phobert-base"
MAX_LEN, EPOCHS, WARMUP = 256, 3, 0.06
# Training batch: 256 in the manuscript (used by G3). G2 may be run with a larger
# batch (TRAIN_BATCH=512) for speed; both G2 runs then use the same value, so the
# head vs head+tail comparison stays like-for-like. EVAL_BATCH only affects speed.
BATCH = int(os.environ.get("TRAIN_BATCH", 256))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", 1024))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Learning rate for the PhoBERT reruns. The released PhoBERT trainer_state.json logs
# 2.1045e-06 at step 8000 of 8877 with 6% warm-up, which corresponds to a peak of 2e-5.
# Using the same value lets the G4 baseline reproduce the Table 2 PhoBERT result.
PHOBERT_LR = float(os.environ.get("PHOBERT_LR", 2e-5))
SEED = 42
CAFE_EXPECTED_CORRECT = {42: 89885, 22: 89848}   # from all_seeds_metrics_cafebert.json
STAGES = sys.argv[1:] or ["G1", "G2", "G3", "SUMMARY"]
dev = "cuda"
log_lines = []


def log(msg=""):
    print(msg, flush=True)
    log_lines.append(str(msg))
    (OUT / "step4_report.txt").write_text("\n".join(log_lines), encoding="utf-8")


def metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(n=int(len(y)), accuracy=float(np.mean(pred == y)), macro_precision=float(p),
                macro_recall=float(r), macro_f1=float(f),
                macro_auc=float(roc_auc_score(y, prob, multi_class="ovr", average="macro")))


def reference_true():
    return np.load(pred_path("PhoBERT", 42))["true"].astype(int)


def load_split(repo, split):
    ds = load_dataset(repo, split=split)
    return ds


def label_offset(test_labels):
    true = reference_true()
    for k in (0, 1):
        if np.array_equal(np.asarray(test_labels) - k, true):
            return k
    sys.exit("dataset test labels are not in prediction-file order")


NPROC = os.cpu_count()


def tokenize_all(tok, texts, mode, labels=None):
    """Tokenize in parallel with NPROC worker processes (the PhoBERT tokenizer is slow)."""
    data = {"text": list(texts)}
    if labels is not None:
        data["labels"] = list(labels)
    return Dataset.from_dict(data).map(
        lambda b: {"input_ids": encode(tok, b["text"], mode)}, batched=True, batch_size=1000,
        num_proc=NPROC, remove_columns=["text"])


@torch.no_grad()
def predict(model, tok, texts, mode="head"):
    model.eval().to(dev)
    ids = tokenize_all(tok, texts, mode)["input_ids"]
    order = np.argsort([len(x) for x in ids])          # length-sorted batches: less padding
    probs = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(ids), EVAL_BATCH):
        idx = order[i:i + EVAL_BATCH]
        batch = tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        probs[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    return probs


def encode(tok, texts, mode):
    if mode == "head":
        return tok(texts, truncation=True, max_length=MAX_LEN)["input_ids"]
    budget = MAX_LEN - tok.num_special_tokens_to_add(pair=False)
    head = budget // 2
    out = []
    for ids in tok(texts, add_special_tokens=False, truncation=False)["input_ids"]:
        if len(ids) > budget:
            ids = ids[:head] + ids[-(budget - head):]
        out.append(tok.build_inputs_with_special_tokens(ids))
    return out


def finetune(name, train_texts, train_y, dev_texts, dev_y, mode):
    """Fine-tune PhoBERT with the manuscript protocol; returns (model, tok, seconds)."""
    tok = AutoTokenizer.from_pretrained(PHOBERT)

    tr = tokenize_all(tok, train_texts, mode, train_y)
    dv = tokenize_all(tok, dev_texts, mode, dev_y)
    log(f"{name}: tokenized {len(tr)} train / {len(dv)} dev articles with {NPROC} processes")
    model = AutoModelForSequenceClassification.from_pretrained(PHOBERT, num_labels=10)
    steps = math.ceil(len(tr) / BATCH) * EPOCHS

    def compute(ev):
        logits, y = ev
        e = np.exp(logits - logits.max(1, keepdims=True))
        return {k: v for k, v in metrics(y, e / e.sum(1, keepdims=True)).items() if k != "n"}
    args = TrainingArguments(
        output_dir=f"/content/step4_ckpt/{name}",  # local disk, not Drive
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=EVAL_BATCH, num_train_epochs=EPOCHS, learning_rate=PHOBERT_LR,
        optim="adamw_torch_fused", dataloader_pin_memory=True,
        warmup_steps=math.ceil(WARMUP * steps), lr_scheduler_type="linear", bf16=True,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        seed=SEED, logging_steps=500, report_to="none", disable_tqdm=True,
        dataloader_num_workers=min(8, os.cpu_count()))
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=dv,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    t0 = time.time()
    trainer.train()
    return trainer.model, tok, time.time() - t0


def save_run(name, results, extra=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(results=results, **(extra or {})), open(d / "metrics.json", "w"), indent=2)


def done(name):
    return (OUT / name / "metrics.json").exists()


# ================================================================= G1 CafeBERT
def stage_g1():
    test = load_split(CLEAN, "test")
    off = label_offset(test["label"])
    y = np.asarray(test["label"]) - off
    texts = test["content"]
    roots = [os.environ["CAFE_DIR"]] if os.environ.get("CAFE_DIR") else [
        p for depth in ("*", "*/*", "*/*/*") for name in ("CafeBERT_Results", "CafeBERT_RESULTS")
        for p in glob.glob(f"/content/drive/{depth}/{name}")]
    if not roots:
        sys.exit("CafeBERT_Results not found under /content/drive (mount Drive / add a shortcut)")
    for s in (42, 22):
        name = f"G1_cafebert_seed{s}"
        if done(name):
            continue
        ck = sorted(glob.glob(f"{roots[0]}/cafebert_checkpoints_seed_{s}/checkpoint-*"))
        if len(ck) != 1:
            log(f"G1 seed {s}: expected one checkpoint folder, found {ck}")
            continue
        cfg_dir = ck[0]
        if not os.path.exists(f"{ck[0]}/config.json"):
            # the seed-42 folder holds only the weights and tokenizer.json; the seed-22
            # folder holds the config of the same XLMRobertaForSequenceClassification
            cfg_dir = sorted(glob.glob(f"{roots[0]}/cafebert_checkpoints_seed_*/checkpoint-*/config.json"))[0]
            cfg_dir = os.path.dirname(cfg_dir)
            log(f"G1 seed {s}: config.json missing, using {cfg_dir}/config.json")
        from transformers import AutoConfig
        model = AutoModelForSequenceClassification.from_pretrained(
            ck[0], config=AutoConfig.from_pretrained(cfg_dir))
        # The seed-42 folder has tokenizer.json but no tokenizer_config.json, so the
        # special tokens (<pad>, <s>, </s>) are not registered. Use a folder that has
        # both files (same CafeBERT tokenizer), else the public uitnlp/CafeBERT one.
        tok_dirs = [d for d in [ck[0]] + sorted(glob.glob(f"{roots[0]}/cafebert_checkpoints_seed_*/checkpoint-*"))
                    if os.path.exists(f"{d}/tokenizer_config.json")]
        tok = AutoTokenizer.from_pretrained(tok_dirs[0]) if tok_dirs else None
        if tok is None or tok.pad_token is None:
            tok = AutoTokenizer.from_pretrained("uitnlp/CafeBERT")
            log(f"G1 seed {s}: tokenizer loaded from uitnlp/CafeBERT")
        else:
            log(f"G1 seed {s}: tokenizer loaded from {tok_dirs[0]}")
        if tok.pad_token is None or tok.pad_token_id != model.config.pad_token_id:
            sys.exit(f"G1 seed {s}: pad token {tok.pad_token!r}/{tok.pad_token_id} does not match "
                     f"model pad_token_id {model.config.pad_token_id}")
        prob = predict(model, tok, texts)
        correct = int(np.sum(prob.argmax(1) == y))
        log(f"G1 seed {s}: {correct} correct (released JSON: {CAFE_EXPECTED_CORRECT[s]}, "
            f"difference {correct - CAFE_EXPECTED_CORRECT[s]})")
        np.savez_compressed(OUT / f"{name}_pred.npz", true=y, pred=prob.argmax(1), prob=prob)
        save_run(name, metrics(y, prob), dict(correct=correct, expected=CAFE_EXPECTED_CORRECT[s]))
        del model
        torch.cuda.empty_cache()


# ================================================================= G2 head vs head+tail
def stage_g2():
    tr, dv, te = (load_split(SEG, s) for s in ("train", "dev", "test"))
    off = label_offset(te["label"])
    for mode in ("head", "headtail"):
        name = f"G2_phobert_seed{SEED}_{mode}_bs{BATCH}"
        legacy = f"G2_phobert_seed{SEED}_{mode}"  # earlier runs (batch 256) had no suffix
        if done(name) or (BATCH == 256 and done(legacy)):
            log(f"G2 {mode} (batch {BATCH}): already done, skipped")
            continue
        model, tok, secs = finetune(name, tr["content"], [l - off for l in tr["label"]],
                                    dv["content"], [l - off for l in dv["label"]], mode)
        y = np.asarray(te["label"]) - off
        prob = predict(model, tok, te["content"], mode)
        np.savez_compressed(OUT / f"{name}_pred.npz", true=y, pred=prob.argmax(1), prob=prob)
        save_run(name, metrics(y, prob), dict(train_seconds=secs, lr=PHOBERT_LR, train_batch=BATCH))
        log(f"G2 {mode} (batch {BATCH}, {secs/60:.1f} min): {metrics(y, prob)}")
        del model
        torch.cuda.empty_cache()


# ================================================================= G3 temporal
def stage_g3():
    name = f"G3_phobert_seed{SEED}_temporal"
    if done(name):
        return
    ci = pd.read_pickle(DATA / "corpus_index.pkl")
    sp = {s: load_split(SEG, s) for s in ("train", "dev", "test")}
    off = label_offset(sp["test"]["label"])
    ci["year"] = pd.to_numeric(ci["year"], errors="coerce")
    log(f"G3: rows without a publication year: {int(ci.year.isna().sum())} (excluded)")

    def rows(mask):
        sub = ci[mask]
        texts, ys = [], []
        for s in ("train", "dev", "test"):
            pos = sub.loc[sub.split == s, "pos"].to_numpy()
            if len(pos):
                part = sp[s].select(pos)
                texts += part["content"]
                ys += [l - off for l in part["label"]]
        return texts, np.asarray(ys)
    early = ci.year.between(2019, 2022)
    trX, trY = rows(early & (ci.split == "train"))
    dvX, dvY = rows(early & (ci.split == "dev"))
    log(f"G3: train {len(trY)} / dev {len(dvY)} articles from 2019-2022")
    model, tok, secs = finetune(name, trX, trY.tolist(), dvX, dvY.tolist(), "head")
    res = {}
    for label, mask in [("2019-2022 (original test split)", early & (ci.split == "test")),
                        ("2023", ci.year == 2023), ("2024", ci.year == 2024), ("2025", ci.year == 2025)]:
        X, Y = rows(mask)
        res[label] = metrics(Y, predict(model, tok, X))
        log(f"G3 {label}: {res[label]}")
    save_run(name, res, dict(train_seconds=secs, lr=PHOBERT_LR, train_batch=BATCH, n_train=len(trY), n_dev=len(dvY)))


# ================================================================= summary
def stage_summary():
    out = []
    pho = {s: np.load(pred_path("PhoBERT", s)) for s in SEEDS}
    rng = np.random.default_rng(20261106)
    for s in (42, 22):
        f = OUT / f"G1_cafebert_seed{s}_pred.npz"
        if not f.exists():
            continue
        c, p = np.load(f), pho[s]
        y = p["true"].astype(int)
        assert np.array_equal(c["true"], y)
        ca, cb = c["pred"] == y, p["pred"] == y
        n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        f1 = lambda yy, pp: precision_recall_fscore_support(yy, pp, average="macro", zero_division=0)[2]
        d_obs = f1(y, c["pred"]) - f1(y, p["pred"])
        boot = []
        for _ in range(10000):
            i = rng.integers(0, len(y), len(y))
            boot.append(f1(y[i], c["pred"][i]) - f1(y[i], p["pred"][i]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(dict(seed=s, delta_macro_f1_pp=100 * d_obs, ci95_low=100 * lo, ci95_high=100 * hi,
                        bootstrap_p=min(1, 2 * min(np.mean(np.array(boot) <= 0), np.mean(np.array(boot) >= 0))),
                        cafe_only_correct=n01, phobert_only_correct=n10,
                        mcnemar_exact_p=binomtest(n01, n01 + n10, 0.5).pvalue))
    if out:
        pd.DataFrame(out).to_csv(OUT / "G1_paired_tests.csv", index=False)
        log("G1 paired tests (CafeBERT - PhoBERT):\n" + pd.DataFrame(out).round(4).to_string(index=False))
    for f in sorted(OUT.glob("G*/metrics.json")):
        log(f"{f.parent.name}: {json.dumps(json.load(open(f))['results'])[:600]}")


for st in STAGES:
    log(f"===== {st}")
    {"G1": stage_g1, "G2": stage_g2, "G3": stage_g3, "SUMMARY": stage_summary}[st]()
log("Done. Send the folder " + str(OUT) + ".")
