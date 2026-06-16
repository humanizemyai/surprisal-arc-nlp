"""Does the FULL surprisal band-energy distribution (DJ-EQ / E(tau)) carry human-vs-AI signal
that the single low-band arc-shape missed?

Tests the 8-band relative-power distribution of the surprisal spectrum (global arc -> token jitter)
against arc-shape (null in prior experiments) and the mean+burstiness baseline.
HC3 + MAGE raw + MAGE paraphrased, gpt2 surprisal, 5-fold ROC-AUC.

Run: .venv/bin/python src/detect/band_energy_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.arc.features import arc_shape_features, band_energy_features, baseline_features  # noqa: E402
from src.data.load_data import load_hc3, load_mage_ood  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals  # noqa: E402


def cv_auc(X, y, folds=5, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return round(float(cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean()), 3)


def featurize(texts, model, tok, dev, max_tokens):
    from tqdm import tqdm
    base, arc, band = [], [], []
    for t in texts:
        s = token_surprisals(t, model, tok, dev, max_tokens)
        b = baseline_features(s)
        base.append([b["mean"], b["std"], b["mad"]])
        arc.append(arc_shape_features(s))
        band.append(band_energy_features(s))
    return np.array(base), np.array(arc), np.array(band)


def report(name, base, arc, band, y):
    print(f"  {name}")
    print(f"      mean+burst   {cv_auc(base, y)}")
    print(f"      arc_shape    {cv_auc(arc, y)}")
    print(f"      band_energy  {cv_auc(band, y)}")
    print(f"      band+base    {cv_auc(np.hstack([base, band]), y)}")


def main():
    from tqdm import tqdm  # noqa: F401
    model, tok, dev = load_lm("gpt2", "auto")
    print(f"device={dev}\nHuman-vs-AI 5-fold ROC-AUC (gpt2 surprisal):\n")

    texts, labels = load_hc3("open_qa", 150, 40, 42)
    b, a, bd = featurize(texts, model, tok, dev, 512)
    report("HC3 (raw ChatGPT)", b, a, bd, np.array(labels))

    pools = load_mage_ood(min_words=60, max_per_cell=300, seed=42)
    h = pools["human"]
    bh, ah, bdh = featurize(h, model, tok, dev, 768)
    br, ar, bdr = featurize(pools["ai_raw"], model, tok, dev, 768)
    bp, ap, bdp = featurize(pools["ai_para"], model, tok, dev, 768)
    yr = np.array([0] * len(h) + [1] * len(pools["ai_raw"]))
    yp = np.array([0] * len(h) + [1] * len(pools["ai_para"]))
    report("MAGE raw (human vs GPT4)", np.vstack([bh, br]), np.vstack([ah, ar]), np.vstack([bdh, bdr]), yr)
    report("MAGE paraphrased (attack)", np.vstack([bh, bp]), np.vstack([ah, ap]), np.vstack([bdh, bdp]), yp)


if __name__ == "__main__":
    main()
