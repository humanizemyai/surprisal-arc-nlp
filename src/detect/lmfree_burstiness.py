"""LM-free burstiness as a /detect signal.

Does sentence-length variation (std / coefficient-of-variation / mean-adjacent-diff) separate
human vs AI on HC3 + MAGE raw + MAGE paraphrased? If it survives the paraphrase attack, it's a
shippable, attack-robust feature for the TS detector (no LM needed at runtime).

Run: .venv/bin/python src/detect/lmfree_burstiness.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data.load_data import load_hc3, load_mage_ood  # noqa: E402


def sent_word_lengths(text: str) -> np.ndarray:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sents]
    return np.array([x for x in lengths if x > 0], float)


def lmfree_feats(text: str) -> dict:
    """LM-free burstiness features the TS detector can compute."""
    L = sent_word_lengths(text)
    if L.size < 2:
        return {"sl_std": 0.0, "sl_cv": 0.0, "sl_mad": 0.0}
    mean = L.mean()
    return {
        "sl_std": float(L.std()),                                   # current detector uses this (binary <4.2)
        "sl_cv": float(L.std() / mean) if mean > 0 else 0.0,        # scale-free burstiness
        "sl_mad": float(np.mean(np.abs(np.diff(L)))),               # local roughness
    }


def disc_auc(scores, y) -> float:
    """Direction-agnostic separation AUC (humans are more bursty -> feature lower for AI)."""
    a = roc_auc_score(y, scores)
    return max(a, 1 - a)


def report(name, texts, y):
    feats = [lmfree_feats(t) for t in texts]
    print(f"  {name:28s} (n={len(texts)})")
    for k in ("sl_std", "sl_cv", "sl_mad"):
        a = disc_auc([f[k] for f in feats], y)
        print(f"      {k:8s} AUC = {a:.3f}")


def main():
    print("LM-free sentence-length burstiness — human(0) vs AI(1) separation AUC\n")
    # E1 HC3
    texts, labels = load_hc3("open_qa", 200, 40, 42)
    report("HC3 (raw ChatGPT)", texts, np.array(labels))

    # E2 MAGE
    pools = load_mage_ood(min_words=60, max_per_cell=300, seed=42)
    h = pools["human"]
    report("MAGE raw (human vs GPT4)", h + pools["ai_raw"],
           np.array([0] * len(h) + [1] * len(pools["ai_raw"])))
    report("MAGE paraphrased (attack)", h + pools["ai_para"],
           np.array([0] * len(h) + [1] * len(pools["ai_para"])))


if __name__ == "__main__":
    main()
