"""STEP 5 - Does the end of an article contain the opening of ANOTHER article?
(check before interpreting the head+tail gain of Reviewer #2, Concern 3)

For a random sample of articles (default 200, seed 2026) the last TAIL_WORDS words
are compared with the first LEAD_WORDS words (title + lead) of every other article
in BaoNhan/Vietnam_News_3_Sources_Plus, using 8-word shingles. A sampled article
"contains a teaser" when its tail shares >= MIN_HITS shingles with the opening of
another article that is NOT a duplicate of the sampled article itself.

Outputs (revision_results/): tail_teaser_sample.csv (one row per sampled article,
with the tail text for manual reading) and tail_teaser_report.txt.
CPU (High-RAM), about 10-15 minutes. Set N_SAMPLE=5000 for a tighter estimate.
"""
import os
import random
import re
import zlib
from pathlib import Path

import pandas as pd

K, TAIL_WORDS, LEAD_WORDS, MIN_HITS = 8, 60, 60, 3
SEED = 2026
WS = re.compile(r"\s+")
OUT = Path("revision_results")


def words(text, limit_chars=None):
    t = text or ""
    if limit_chars:
        t = t[:limit_chars]
    return WS.sub(" ", t).strip().lower().split(" ")


def shingles(ws):
    return {zlib.crc32(" ".join(ws[i:i + K]).encode("utf-8"))
            for i in range(max(0, len(ws) - K + 1))}


def tail_words(text):
    w = words(text)
    if len(w) < TAIL_WORDS + LEAD_WORDS:      # short article: tail = second half only
        return w[len(w) // 2:]
    return w[-TAIL_WORDS:]


def find_teasers(contents, sample_rows, progress=None):
    """contents: indexable sequence of article texts (whole corpus).
    Returns {row: (best_other_row or None, hits, lead_overlap_with_other)}."""
    tails = {r: shingles(tail_words(contents[r])) for r in sample_rows}
    leads = {r: shingles(words(contents[r], 3000)[:LEAD_WORDS]) for r in sample_rows}
    want = {}
    for r, sh in tails.items():
        for h in sh:
            want.setdefault(h, []).append(r)
    hits = {r: {} for r in sample_rows}
    n = len(contents)
    for other in range(n):
        for h in shingles(words(contents[other], 3000)[:LEAD_WORDS]):
            for r in want.get(h, ()):
                if r != other:
                    hits[r][other] = hits[r].get(other, 0) + 1
        if progress and other % 100000 == 0:
            progress(other, n)
    out = {}
    for r in sample_rows:
        cands = sorted(hits[r].items(), key=lambda kv: -kv[1])
        best = None
        for other, c in cands:
            if c < MIN_HITS:
                break
            overlap = len(leads[r] & shingles(words(contents[other], 3000)[:LEAD_WORDS]))
            if overlap >= MIN_HITS:          # same opening -> duplicate article, not a teaser
                continue
            best = (other, c, overlap)
            break
        out[r] = best if best else (None, cands[0][1] if cands else 0, 0)
    return out


if __name__ == "__main__":
    from datasets import load_dataset
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("BaoNhan/Vietnam_News_3_Sources_Plus", split="train")
    contents = ds["content"]
    meta = ds.remove_columns(["content"]).to_pandas()
    n_sample = os.environ.get("N_SAMPLE", "200")
    rows = sorted(random.Random(SEED).sample(range(len(ds)), int(n_sample)))
    res = find_teasers(contents, rows, progress=lambda i, n: print(f"  scanned {i}/{n}", flush=True))
    recs = []
    for r in rows:
        other, h, ov = res[r]
        rec = dict(row=r, source=meta.source[r], label=meta.label[r], url=meta.url[r],
                   has_teaser=other is not None, shared_shingles=h,
                   other_label=meta.label[other] if other is not None else None,
                   other_url=meta.url[other] if other is not None else None,
                   tail_text=" ".join(tail_words(contents[r])))
        rec["same_label"] = rec["has_teaser"] and rec["other_label"] == rec["label"]
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tail_teaser_sample.csv", index=False)
    lines = [f"sampled articles: {len(df)} (seed {SEED}); tail = last {TAIL_WORDS} words, "
             f"lead = first {LEAD_WORDS} words, {K}-word shingles, >= {MIN_HITS} shared",
             f"tail contains the opening of another article: {df.has_teaser.sum()} "
             f"({100 * df.has_teaser.mean():.1f}%)"]
    t = df[df.has_teaser]
    if len(t):
        lines.append(f"  ... and that article has the SAME category: {t.same_label.sum()} "
                     f"({100 * t.same_label.mean():.1f}% of those)")
    lines.append("by source (share with teaser):")
    lines.append(df.groupby("source").has_teaser.agg(["size", "mean"]).round(3).to_string())
    lines.append("by category (share with teaser):")
    lines.append(df.groupby("label").has_teaser.agg(["size", "mean"]).round(3).to_string())
    (OUT / "tail_teaser_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nTen examples with a teaser (end of article):")
    for _, x in t.head(10).iterrows():
        print(f"- [{x.label} -> {x.other_label}] ...{x.tail_text[-220:]}")
