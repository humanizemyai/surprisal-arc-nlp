"""Robustness figure: arc null + perplexity-collapse/burstiness-hold across two surprisal LMs.
Reads tables/results_*.json (no model needed). Usage: .venv/bin/python src/viz/robustness_fig.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(lm):
    return json.load(open(os.path.join(ROOT, "tables", f"results_{lm}.json")))


g, p = load("gpt2"), load("EleutherAI_pythia-410m")
w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

# Panel A — arc-shape at chance across conditions, both LMs
conds = ["E1_hc3", "E2_mage_raw", "E2_mage_para"]
labels = ["HC3", "MAGE raw", "MAGE para"]
xi = np.arange(len(conds))
ax[0].bar(xi - w / 2, [g[c]["arc"] for c in conds], w, label="gpt2", color="#A855F7")
ax[0].bar(xi + w / 2, [p[c]["arc"] for c in conds], w, label="pythia-410m", color="#7C3AED")
ax[0].axhline(0.5, color="#888", ls="--", lw=0.8)
ax[0].set_ylim(0.4, 1.0); ax[0].set_xticks(xi); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("ROC-AUC"); ax[0].set_title("Arc-shape is at chance (both LMs)"); ax[0].legend()

# Panel B — mean & burstiness, raw vs paraphrased, both LMs
groups = ["mean\nraw", "mean\npara", "burst\nraw", "burst\npara"]
gv = [g["E2_mage_raw"]["mean"], g["E2_mage_para"]["mean"], g["E2_mage_raw"]["burst"], g["E2_mage_para"]["burst"]]
pv = [p["E2_mage_raw"]["mean"], p["E2_mage_para"]["mean"], p["E2_mage_raw"]["burst"], p["E2_mage_para"]["burst"]]
xi2 = np.arange(4)
ax[1].bar(xi2 - w / 2, gv, w, label="gpt2", color="#60A5FA")
ax[1].bar(xi2 + w / 2, pv, w, label="pythia-410m", color="#2563EB")
ax[1].axhline(0.5, color="#888", ls="--", lw=0.8)
ax[1].set_ylim(0.4, 1.0); ax[1].set_xticks(xi2); ax[1].set_xticklabels(groups)
ax[1].set_ylabel("ROC-AUC"); ax[1].set_title("Paraphrase collapses perplexity, not burstiness"); ax[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "robustness_two_lms.png"), dpi=140)
print("saved figures/robustness_two_lms.png")
