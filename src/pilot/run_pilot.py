"""PILOT GATE: does surprisal arc-SHAPE beat burstiness at human-vs-AI separation?

Loads a balanced HC3 sample (human vs ChatGPT), extracts per-token surprisal with gpt2,
builds baseline (mean/burstiness) vs arc-shape (FFT) features, and compares 5-fold
ROC-AUC. Pass if arc-shape AUC materially beats burstiness (target >= 0.65 vs ~0.50).

Run:  .venv/bin/python src/pilot/run_pilot.py
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
from src.data.load_data import load_hc3  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals  # noqa: E402


def cv_auc(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main() -> None:
    with open(os.path.join(ROOT, "config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    pc, ac, dc = cfg["pilot"], cfg["arc"], cfg["detect"]
    seed = pc["seed"]
    np.random.seed(seed)

    print(f"[1/4] Loading HC3 ({pc['config']}, {pc['n_per_class']}/class, >= {pc['min_words']} words)...")
    texts, labels = load_hc3(pc["config"], pc["n_per_class"], pc["min_words"], seed)
    y = np.array(labels)
    print(f"      got {int((y == 0).sum())} human + {int((y == 1).sum())} AI texts")

    model_name = cfg["surprisal"]["models"][0]
    print(f"[2/4] Loading LM '{model_name}' + extracting surprisal...")
    model, tok, dev = load_lm(model_name, cfg["surprisal"]["device"])
    print(f"      device = {dev}")

    base_rows, arc_rows, curves = [], [], []
    from tqdm import tqdm
    for t in tqdm(texts, ncols=80):
        s = token_surprisals(t, model, tok, dev, cfg["surprisal"]["max_tokens"])
        b = baseline_features(s)
        base_rows.append([b["mean"], b["std"], b["mad"]])
        arc_rows.append(arc_shape_features(s, ac["resample_len"], ac["smooth_window"], ac["n_fft_coeffs"]))
        curves.append(normalized_arc(s, ac["resample_len"], ac["smooth_window"]))

    base = np.array(base_rows)          # [mean, std, mad]
    arc = np.array(arc_rows)            # FFT low-freq magnitudes
    X_mean = base[:, [0]]
    X_burst = base[:, [1, 2]]          # std + local variation = burstiness
    X_arc = arc
    X_all = np.hstack([base, arc])

    print("[3/4] 5-fold logistic-regression ROC-AUC...")
    results = {}
    for name, X in [("mean_perplexity", X_mean), ("burstiness", X_burst),
                    ("arc_shape", X_arc), ("all", X_all)]:
        m, sd = cv_auc(X, y, dc["cv_folds"], seed)
        results[name] = {"auc_mean": round(m, 4), "auc_std": round(sd, 4)}
        print(f"      {name:16s} AUC = {m:.3f} ± {sd:.3f}")

    arc_auc = results["arc_shape"]["auc_mean"]
    burst_auc = results["burstiness"]["auc_mean"]
    passed = (arc_auc >= 0.65) and (arc_auc > burst_auc + 0.03)
    verdict = "PASS" if passed else "FAIL"
    print(f"\n[GATE] arc_shape {arc_auc:.3f} vs burstiness {burst_auc:.3f}  ->  {verdict}")

    out = {
        "dataset": f"HC3/{pc['config']}", "lm": model_name, "device": dev,
        "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
        "config": {"arc": ac, "detect": dc, "min_words": pc["min_words"]},
        "results": results,
        "gate": {"arc_shape": arc_auc, "burstiness": burst_auc, "passed": passed},
    }
    os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "tables", "pilot_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("[4/4] Saving figure...")
    _save_figure(np.array(curves), y, results, model_name)
    print("      -> tables/pilot_results.json + figures/pilot_arc_vs_burstiness.png")


def _save_figure(curves: np.ndarray, y: np.ndarray, results: dict, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    h_mean, a_mean = curves[y == 0].mean(0), curves[y == 1].mean(0)
    x = np.linspace(0, 1, curves.shape[1])
    ax[0].plot(x, h_mean, label="human", color="#1B9E4B", lw=2)
    ax[0].plot(x, a_mean, label="AI (ChatGPT)", color="#EC4899", lw=2)
    ax[0].axhline(0, color="#888", lw=0.6, ls="--")
    ax[0].set_title("Mean normalized surprisal arc T(t)")
    ax[0].set_xlabel("normalized position in text")
    ax[0].set_ylabel("z-surprisal")
    ax[0].legend()

    names = list(results.keys())
    aucs = [results[n]["auc_mean"] for n in names]
    errs = [results[n]["auc_std"] for n in names]
    colors = ["#9CA3AF", "#9CA3AF", "#A855F7", "#60A5FA"]
    ax[1].bar(names, aucs, yerr=errs, color=colors, capsize=4)
    ax[1].axhline(0.5, color="#888", lw=0.8, ls="--", label="chance")
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title(f"Human-vs-AI ROC-AUC (LM={model_name})")
    ax[1].set_ylabel("ROC-AUC (5-fold)")
    ax[1].tick_params(axis="x", rotation=20)
    ax[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "pilot_arc_vs_burstiness.png"), dpi=140)


if __name__ == "__main__":
    main()
