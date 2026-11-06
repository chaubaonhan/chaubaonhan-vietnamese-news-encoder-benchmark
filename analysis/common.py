"""Shared configuration for the PLOS ONE revision analyses (PONE-D-26-38656).

Every constant below is taken from the manuscript or from the Hugging Face
repositories listed by step0_inspect.py. Nothing here is estimated.
"""
from pathlib import Path

SEEDS = [42, 22, 202]
FIG_SEED_ORDER = [22, 42, 202]          # panel order of the original Fig 5
MAX_LEN = 256                           # model-specific token budget
N_CLASSES = 10

# Model display name -> (HF repo, short key used in folder/file names)
MODELS = {
    "WikiBERT":   ("BaoNhan/WikiBERT_Vietnamese_News_Classification", "wikibert"),
    "PhoBERT":    ("BaoNhan/PhoBERT_Vietnamese_News_Classification", "phobert"),
    "mBERT":      ("BaoNhan/mBERT_Vietnamese_News_Classification", "mbert"),
    "DistilBERT": ("BaoNhan/DistilBERT_Vietnamese_News_Classification", "distilbert"),
    "XLM-R":      ("BaoNhan/XLMR_Vietnamese_News_Classification", "xlmr"),
    "CafeBERT":   (None, "cafebert"),   # fine-tuned CafeBERT runs are not on the Hub
    "viBERT":     ("BaoNhan/viBERT_Vietnamese_News_Classification", "vibert"),
    "vELECTRA":   ("BaoNhan/vELECTRA_Vietnamese_News_Classification", "velectra"),
    "ViSoBERT":   ("BaoNhan/ViSoBERT_Vietnamese_News_Classification", "visobert"),
}
MODEL_ORDER = list(MODELS)  # same order as Table 2
# Public pre-trained checkpoints that were fine-tuned (verified to exist on the Hub;
# step4 also checks their sizes against the released fine-tuned configs).
BASE_CHECKPOINTS = {
    "WikiBERT":   "TurkuNLP/wikibert-base-vi-cased",
    "PhoBERT":    "vinai/phobert-base",
    "mBERT":      "google-bert/bert-base-multilingual-cased",
    "DistilBERT": "distilbert/distilbert-base-multilingual-cased",
    "XLM-R":      "FacebookAI/xlm-roberta-base",
    "CafeBERT":   "uitnlp/CafeBERT",
    "viBERT":     "FPTAI/vibert-base-cased",
    "vELECTRA":   "FPTAI/velectra-base-discriminator-cased",
    "ViSoBERT":   "uitnlp/visobert",
}
# Models whose seed-specific TCC was not retained: Fig 5 uses the Table 2 mean TCC
# for every seed of these models and the caption states this explicitly.
MEAN_TCC_FALLBACK = ["PhoBERT", "CafeBERT"]

# Category indices used by all prediction files (manuscript, Results section)
LABEL_NAMES = ["Education", "Entertainment", "Business", "Law", "Health",
               "World", "Sports", "Politics", "Automotive", "Lifestyle"]
# Test-set row totals per index, as reported in the manuscript
TEST_COUNTS = [6090, 9542, 12350, 6992, 9766, 13673, 12423, 13490, 4844, 5500]
N_TEST = 94670

# Table 2 of the submitted manuscript: mean (population SD) over seeds
TABLE2 = {  # loss, acc, P, R, F1, AUC, TCC
    "WikiBERT":   [0.20330, 0.93016, 0.92377, 0.92264, 0.92302, 0.99635, 3077.08],
    "PhoBERT":    [0.14761, 0.94779, 0.94124, 0.94375, 0.94241, 0.99794, 3360.89],
    "mBERT":      [0.16421, 0.94270, 0.93707, 0.93660, 0.93674, 0.99750, 3181.59],
    "DistilBERT": [0.17685, 0.93793, 0.93145, 0.93213, 0.93167, 0.99713, 1554.86],
    "XLM-R":      [0.15650, 0.94488, 0.93907, 0.94011, 0.93953, 0.99770, 3099.05],
    "CafeBERT":   [0.14729, 0.94916, 0.94391, 0.94445, 0.94413, 0.99800, 9335.51],
    "viBERT":     [0.20287, 0.92903, 0.92226, 0.92185, 0.92190, 0.99629, 3065.10],
    "vELECTRA":   [0.18201, 0.93791, 0.93174, 0.93189, 0.93161, 0.99710, 3192.59],
    "ViSoBERT":   [0.21663, 0.92617, 0.91819, 0.92075, 0.91934, 0.99604, 3201.82],
}
TABLE2_TCC_SD = {"WikiBERT": 3.46492, "PhoBERT": 4.53214, "mBERT": 1.24008,
                 "DistilBERT": 0.99322, "XLM-R": 1.41930, "CafeBERT": 148.46973,
                 "viBERT": 1.53178, "vELECTRA": 1.57443, "ViSoBERT": 0.44646}

COLORS = {"WikiBERT": "#4C78A8", "PhoBERT": "#F58518", "mBERT": "#54A24B",
          "DistilBERT": "#E45756", "XLM-R": "#B279A2", "CafeBERT": "#72B7B2",
          "viBERT": "#FF9DA6", "vELECTRA": "#9D755D", "ViSoBERT": "#BAB0AC"}

DATA = Path("revision_data")      # created by step1
OUT = Path("revision_results")    # created by step2/step3


def pred_path(model, seed):
    return DATA / "preds" / f"{model}_seed{seed}.npz"
