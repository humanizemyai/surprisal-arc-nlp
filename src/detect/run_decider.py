"""DECIDER: does surprisal arc-SHAPE add signal WHERE mean+burstiness collapse?

Long-form CNN-news (MAGE GPT4 OOD), two conditions:
  A (raw)  : human vs GPT4
  B (para) : human vs GPT4 *paraphrased* (the adversarial / humanized condition)

For each condition, 5-fold ROC-AUC for mean(perplexity) / burstiness / arc-shape / mean+burst / all.
Headline: if mean+burst AUC drops sharply A->B but arc-shape holds (and adds over mean+burst in B),
the surprisal arc is a real, attack-robust signal. If arc stays ~chance, pivot to the UID-overshoot paper.

Run:  .venv/bin/python src/detect/run_decider.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.arc.features import arc_shape_features, baseline_features, normalized_arc  # noqa: E402
from src.data.load_data import load_mage_ood  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals  # noqa: E402

MAX_TOKENS = 768  # CNN news runs long; gpt2 context = 1024


def cv_auc(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    s = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def featurize(texts, model, tok, dev, ac, cache):
    base, arc, curves = [], [], []
    for t in texts:
        s = cache.get(t)
        if s is None:
            s = token_surprisals(t, model, tok, dev, MAX_TOKENS)
            cache[t] = s
        b = baseline_features(s)
        base.append([b["mean"], b["std"], b["mad"]])
        arc.append(arc_shape_features(s, ac["resample_len"], ac["smooth_window"], ac["n_fft_coeffs"]))
        curves.append(normalized_arc(s, ac["resample_len"], ac["smooth_window"]))
    return np.array(base), np.array(arc), np.array(curves)


def condition_aucs(base_h, arc_h, base_ai, arc_ai, folds, seed) -> dict:
    Xb = np.vstack([base_h, base_ai])
    Xa = np.vstack([arc_h, arc_ai])
    y = np.array([0] * len(base_h) + [1] * len(base_ai))
    sets = {
        "mean_perplexity": Xb[:, [0]],
        "burstiness": Xb[:, [1, 2]],
        "arc_shape": Xa,
        "mean+burst": Xb,
        "all": np.hstack([Xb, Xa]),
    }
    out = {}
    for k, X in sets.items():
        m, sd = cv_auc(X, y, folds, seed)
        out[k] = {"auc_mean": round(m, 4), "auc_std": round(sd, 4)}
    return out


def main() -> None:
    with open(os.path.join(ROOT, "config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ac, dc = cfg["arc"], cfg["detect"]
    seed = dc["seed"]
    np.random.seed(seed)

    print("[1/4] Loading MAGE GPT4 OOD (raw + paraphrased), domain-matched...")
    pools = load_mage_ood(min_words=60, max_per_cell=250, seed=seed)
    print(f"      domains={pools['domains']} | human={len(pools['human'])} "
          f"ai_raw={len(pools['ai_raw'])} ai_para={len(pools['ai_para'])}")

    model_name = cfg["surprisal"]["models"][0]
    print(f"[2/4] Loading LM '{model_name}' + extracting surprisal...")
    model, tok, dev = load_lm(model_name, cfg["surprisal"]["device"])
    print(f"      device={dev}")

    cache: dict = {}
    bh, ah, ch = featurize(pools["human"], model, tok, dev, ac, cache)
    bra, ara, cra = featurize(pools["ai_raw"], model, tok, dev, ac, cache)
    bpa, apa, cpa = featurize(pools["ai_para"], model, tok, dev, ac, cache)

    print("[3/4] 5-fold ROC-AUC per condition...")
    res = {
        "A_raw  (human vs GPT4)": condition_aucs(bh, ah, bra, ara, dc["cv_folds"], seed),
        "B_para (human vs GPT4-paraphrased)": condition_aucs(bh, ah, bpa, apa, dc["cv_folds"], seed),
    }
    feats = ["mean_perplexity", "burstiness", "arc_shape", "mean+burst", "all"]
    print(f"\n      {'condition':36s} " + " ".join(f"{f:>14s}" for f in feats))
    for cond, d in res.items():
        print(f"      {cond:36s} " + " ".join(f"{d[f]['auc_mean']:>14.3f}" for f in feats))

    a, b = res["A_raw  (human vs GPT4)"], res["B_para (human vs GPT4-paraphrased)"]
    drop_mb = a["mean+burst"]["auc_mean"] - b["mean+burst"]["auc_mean"]
    arc_b = b["arc_shape"]["auc_mean"]
    arc_adds = b["all"]["auc_mean"] - b["mean+burst"]["auc_mean"]
    print(f"\n[VERDICT] paraphrase drops mean+burst AUC by {drop_mb:.3f}; "
          f"arc_shape(B)={arc_b:.3f}; arc adds {arc_adds:+.3f} over mean+burst in B")
    if arc_b >= 0.65 and arc_adds >= 0.03:
        print("          -> ARC IS A REAL, ATTACK-ROBUST SIGNAL  (write 'Surprisal Arc' paper)")
    else:
        print("          -> arc still weak  (pivot to 'AI overshoots UID' paper)")

    out = {"dataset": "MAGE/GPT4-OOD (CNN)", "lm": model_name, "device": dev,
           "n": {k: len(pools[k]) for k in ("human", "ai_raw", "ai_para")},
           "domains": pools["domains"], "results": res,
           "summary": {"mean_burst_drop_raw_to_para": round(drop_mb, 4),
                       "arc_auc_para": arc_b, "arc_increment_para": round(arc_adds, 4)}}
    with open(os.path.join(ROOT, "tables", "decider_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("[4/4] Saving figure...")
    _save_figure(ch, cra, cpa, res, model_name)
    print("      -> tables/decider_results.json + figures/decider_arc_paraphrase.png")


def _save_figure(ch, cra, cpa, res, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.linspace(0, 1, ch.shape[1])
    ax[0].plot(x, ch.mean(0), label="human", color="#1B9E4B", lw=2)
    ax[0].plot(x, cra.mean(0), label="GPT4 (raw)", color="#60A5FA", lw=2)
    ax[0].plot(x, cpa.mean(0), label="GPT4 (paraphrased)", color="#EC4899", lw=2)
    ax[0].axhline(0, color="#888", lw=0.6, ls="--")
    ax[0].set_title("Mean normalized surprisal arc T(t) — CNN news")
    ax[0].set_xlabel("normalized position"); ax[0].set_ylabel("z-surprisal"); ax[0].legend()

    feats = ["mean_perplexity", "burstiness", "arc_shape", "mean+burst", "all"]
    conds = list(res.keys())
    w = 0.38
    xi = np.arange(len(feats))
    for j, cond in enumerate(conds):
        vals = [res[cond][f]["auc_mean"] for f in feats]
        errs = [res[cond][f]["auc_std"] for f in feats]
        ax[1].bar(xi + (j - 0.5) * w, vals, w, yerr=errs, capsize=3,
                  label=cond.split()[0], color=["#A855F7", "#EC4899"][j])
    ax[1].axhline(0.5, color="#888", lw=0.8, ls="--")
    ax[1].set_xticks(xi); ax[1].set_xticklabels(feats, rotation=20, ha="right")
    ax[1].set_ylim(0.4, 1.0); ax[1].set_ylabel("ROC-AUC (5-fold)")
    ax[1].set_title(f"Raw vs Paraphrased detection (LM={model_name})"); ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "decider_arc_paraphrase.png"), dpi=140)


if __name__ == "__main__":
    main()
