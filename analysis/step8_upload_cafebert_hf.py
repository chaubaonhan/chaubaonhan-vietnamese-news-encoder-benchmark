"""STEP 8 - Publish the fine-tuned CafeBERT checkpoints, results and article-level test
predictions on Hugging Face, in the same layout as the other eight models
(e.g. BaoNhan/PhoBERT_Vietnamese_News_Classification), and add the repository to the
PLOS ONE collection.

Repository: BaoNhan/CafeBERT_Vietnamese_News_Classification (same visibility as the PhoBERT repo)
  cafebert_checkpoints_seed_<s>/checkpoint-*/   from Drive/CafeBERT_Results
  results_cafebert/all_seeds_metrics.json       from Drive/CafeBERT_Results/results_cafebert
  results_cafebert/confusion_matrix_seed_<s>.png
  results_cafebert/predictions_seed_<s>.csv     from PLOS_revision/*/step4/G1_cafebert_seed<s>_pred.npz
                                                (columns as in the other repositories)
"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, add_collection_item

REPO = "BaoNhan/CafeBERT_Vietnamese_News_Classification"
REF = "BaoNhan/PhoBERT_Vietnamese_News_Classification"
COLLECTION = "BaoNhan/plos-one-6a67a6bec70112e23e5e93e5"
PLOS = "/content/drive/MyDrive/PLOS_revision"
STAGE = "/content/cafebert_hf"
api = HfApi()

roots = [os.environ["CAFE_DIR"]] if os.environ.get("CAFE_DIR") else [
    p for depth in ("*", "*/*", "*/*/*") for p in glob.glob(f"/content/drive/{depth}/CafeBERT_Results")]
if not roots:
    sys.exit("STOP: CafeBERT_Results not found under /content/drive")
root = roots[0]
print("CafeBERT_Results:", root)

private = api.model_info(REF).private
api.create_repo(REPO, repo_type="model", private=private, exist_ok=True)
print(f"repository {REPO} ready (private={private})")

# ---- results_cafebert: metrics, confusion matrices, predictions
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(f"{STAGE}/results_cafebert")
res = f"{root}/results_cafebert"
for f in glob.glob(f"{res}/*"):
    name = os.path.basename(f)
    if name == "all_seeds_metrics_cafebert.json":
        name = "all_seeds_metrics.json"
    shutil.copy(f, f"{STAGE}/results_cafebert/{name}")
seeds = []
for f in sorted(glob.glob(f"{PLOS}/*/step4/G1_cafebert_seed*_pred.npz")):
    s = int(f.split("seed")[-1].split("_")[0])
    r = np.load(f)
    prob = r["prob"].astype(np.float32)
    df = pd.DataFrame({"true_label_0_9": r["true"].astype(int), "predicted_label_0_9": r["pred"].astype(int)})
    for k in range(prob.shape[1]):
        df[f"prob_class_{k}"] = prob[:, k]
    df.to_csv(f"{STAGE}/results_cafebert/predictions_seed_{s}.csv", index=False)
    seeds.append(s)
    print(f"predictions seed {s}: {len(df)} rows, accuracy {np.mean(df.true_label_0_9 == df.predicted_label_0_9):.5f}")
if not seeds:
    sys.exit("STOP: no G1_cafebert_seed*_pred.npz found under PLOS_revision/*/step4")

with open(f"{STAGE}/README.md", "w", encoding="utf-8") as fh:
    fh.write("---\nlanguage: vi\nlicense: mit\nbase_model: uitnlp/CafeBERT\npipeline_tag: text-classification\n"
             "tags:\n- vietnamese\n- news-classification\n---\n\n"
             "# CafeBERT fine-tuned on BN-VN3S (Vietnamese news classification, 10 categories)\n\n"
             "Fine-tuned checkpoints, test-set metrics, confusion matrices and article-level test-set "
             "predictions for the PLOS ONE benchmark of transformer encoders on BN-VN3S.\n\n"
             "* `cafebert_checkpoints_seed_<seed>/` fine-tuned checkpoints and training logs (`trainer_state.json`)\n"
             "* `results_cafebert/all_seeds_metrics.json` test-set metrics per run\n"
             "* `results_cafebert/predictions_seed_<seed>.csv` true label, predicted label and class probabilities "
             "for each of the 94,670 test articles (label indices 0-9 as in the other repositories)\n\n"
             "Code: https://github.com/chaubaonhan/chaubaonhan-vietnamese-news-encoder-benchmark\n")
api.upload_folder(folder_path=STAGE, repo_id=REPO, repo_type="model",
                  commit_message="Add CafeBERT results and article-level test predictions")
print("results uploaded")

# ---- checkpoints
for d in sorted(glob.glob(f"{root}/cafebert_checkpoints_seed_*/checkpoint-*")):
    rel = os.path.relpath(d, root)
    print("uploading", rel, "...")
    api.upload_folder(folder_path=d, path_in_repo=rel, repo_id=REPO, repo_type="model",
                      commit_message=f"Add {rel}")
print("checkpoints uploaded")

add_collection_item(COLLECTION, item_id=REPO, item_type="model", exists_ok=True)
print("added to collection", COLLECTION)
print("Done:", f"https://huggingface.co/{REPO}")
