"""Fine-tune one encoder on BN-VN3S and evaluate it on the held-out test split.

Re-implementation of the fine-tuning protocol described in the paper (Methods):
single fine-tuning run per (model, seed), maximum sequence length 256 model-specific
tokens, AdamW with linear decay after a 6 % warm-up, three epochs, BF16, evaluation
on the development split after every epoch and selection of the epoch with the
highest development macro-F1. All values can be changed from the command line.

Example
    python src/finetune.py --model PhoBERT --seed 42 --out runs/
    python src/finetune.py --model XLM-R --seed 22 --truncation headtail --out runs/

Inputs (private Hugging Face datasets of the authors, access on request):
    BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split   (eight models)
    BaoNhan/Vietnamese_News_PhoBERT_Segmented         (PhoBERT, VnCoreNLP word-segmented)
Outputs in <out>/<model>_seed<seed>/: predictions.csv, metrics.json, trainer logs.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

CHECKPOINTS = {
    "WikiBERT": "TurkuNLP/wikibert-base-vi-cased",
    "PhoBERT": "vinai/phobert-base",
    "mBERT": "google-bert/bert-base-multilingual-cased",
    "DistilBERT": "distilbert/distilbert-base-multilingual-cased",
    "XLM-R": "FacebookAI/xlm-roberta-base",
    "CafeBERT": "uitnlp/CafeBERT",
    "viBERT": "FPTAI/vibert-base-cased",
    "vELECTRA": "FPTAI/velectra-base-discriminator-cased",
    "ViSoBERT": "uitnlp/visobert",
}
CLEAN = "BaoNhan/Vietnamese_News_Cleaned_3_Sources_Split"
SEGMENTED = "BaoNhan/Vietnamese_News_PhoBERT_Segmented"
LABEL_NAMES = ["Education", "Entertainment", "Business", "Law", "Health",
               "World", "Sports", "Politics", "Automotive", "Lifestyle"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(CHECKPOINTS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default="runs")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--truncation", choices=["head", "headtail"], default="head",
                   help="head: first tokens; headtail: first half + last half of the budget")
    p.add_argument("--eval_batch", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=8)
    return p.parse_args()


def encode(tok, texts, mode, max_len):
    if mode == "head":
        return tok(texts, truncation=True, max_length=max_len)["input_ids"]
    budget = max_len - tok.num_special_tokens_to_add(pair=False)
    head = budget // 2
    out = []
    for ids in tok(texts, add_special_tokens=False, truncation=False)["input_ids"]:
        if len(ids) > budget:
            ids = ids[:head] + ids[-(budget - head):]
        out.append(tok.build_inputs_with_special_tokens(ids))
    return out


def metrics(y, prob):
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(accuracy=float(np.mean(pred == y)), macro_precision=float(p), macro_recall=float(r),
                macro_f1=float(f), macro_auc=float(roc_auc_score(y, prob, multi_class="ovr", average="macro")))


def main():
    a = parse()
    run = Path(a.out) / f"{a.model}_seed{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(SEGMENTED if a.model == "PhoBERT" else CLEAN)
    offset = min(ds["train"]["label"])          # dataset labels are stored as 1..10
    tok = AutoTokenizer.from_pretrained(CHECKPOINTS[a.model])

    def prep(split):
        return ds[split].map(
            lambda b: {"input_ids": encode(tok, b["content"], a.truncation, a.max_len),
                       "labels": [l - offset for l in b["label"]]},
            batched=True, batch_size=1000, num_proc=a.num_proc, remove_columns=ds[split].column_names)
    train, dev, test = prep("train"), prep("dev"), prep("test")

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINTS[a.model], num_labels=10)
    steps = math.ceil(len(train) / a.batch) * a.epochs

    def compute(ev):
        logits, y = ev
        e = np.exp(logits - logits.max(1, keepdims=True))
        return metrics(y, e / e.sum(1, keepdims=True))
    args = TrainingArguments(
        output_dir=str(run / "checkpoints"), per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.eval_batch, num_train_epochs=a.epochs, learning_rate=a.lr,
        warmup_steps=math.ceil(a.warmup * steps), lr_scheduler_type="linear", bf16=True,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        seed=a.seed, logging_steps=500, report_to="none", disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    t0 = time.time()
    out = trainer.predict(test)
    predict_s = time.time() - t0

    logits = out.predictions
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    y = np.asarray(test["labels"])
    res = metrics(y, prob)
    res.update(train_duration=train_s, predict_duration=predict_s, tcc=train_s + predict_s,
               settings=vars(a), base_checkpoint=CHECKPOINTS[a.model])
    json.dump(res, open(run / "metrics.json", "w"), indent=2)
    df = pd.DataFrame(prob, columns=[f"prob_class_{k}" for k in range(10)])
    df.insert(0, "predicted_label_0_9", prob.argmax(1))
    df.insert(0, "true_label_0_9", y)
    df.to_csv(run / "predictions.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()
