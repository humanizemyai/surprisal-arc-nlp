"""Run all three experiments for ONE surprisal LM (robustness driver).

E1 HC3 detection, E2 MAGE detection (pooled raw/para + per-domain), E3 ASAP quality.
Lets us confirm the arc null + the UID/attack-robustness positives are not LM-specific.

Usage:
  ARC_LM=gpt2 .venv/bin/python src/run_all.py
  ARC_LM=EleutherAI/pythia-410m .venv/bin/python src/run_all.py
Writes tables/results_<lm>.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.arc.features import arc_shape_features, baseline_features  # noqa: E402
from src.data.load_data import load_asap, load_hc3, load_mage_ood  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals  # noqa: E402


def featurize(texts, model, tok, dev, ac, max_tokens, desc=""):
    from tqdm import tqdm
    base, arc = [], []
    for t in tqdm(texts, ncols=70, desc=desc):
        s = token_surprisals(t, model, tok, dev, max_tokens)
        b = baseline_features(s)
        base.append([b["mean"], b["std"], b["mad"]])
        arc.append(arc_shape_features(s, ac["resample_len"], ac["smooth_window"], ac["n_fft_coeffs"]))
    return np.array(base), np.array(arc)


def cv_auc(X, y, folds, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return round(float(cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean()), 3)


def auc_sets(base, arc, y, folds, seed):
    return {"mean": cv_auc(base[:, [0]], y, folds, seed),
            "burst": cv_auc(base[:, [1, 2]], y, folds, seed),
            "arc": cv_auc(arc, y, folds, seed),
            "mean+burst": cv_auc(base, y, folds, seed)}


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    ac, seed, folds = cfg["arc"], cfg["detect"]["seed"], cfg["detect"]["cv_folds"]
    lm = os.environ.get("ARC_LM", cfg["surprisal"]["models"][0])
    np.random.seed(seed)
    print(f"=== LM = {lm} ===")
    model, tok, dev = load_lm(lm, cfg["surprisal"]["device"])
    res = {"lm": lm, "device": dev}

    # E1 — HC3 detection
    texts, labels = load_hc3("open_qa", 100, 40, seed)
    b, a = featurize(texts, model, tok, dev, ac, 512, "E1 hc3")
    res["E1_hc3"] = auc_sets(b, a, np.array(labels), folds, seed)

    # E2 — MAGE detection (pooled + per-domain)
    pools = load_mage_ood(min_words=60, max_per_cell=300, seed=seed)
    bh, ah = featurize(pools["human"], model, tok, dev, ac, 768, "E2 human")
    bra, ara = featurize(pools["ai_raw"], model, tok, dev, ac, 768, "E2 raw")
    bpa, apa = featurize(pools["ai_para"], model, tok, dev, ac, 768, "E2 para")

    def cond(bx, ax):
        base = np.vstack([bh, bx]); arc = np.vstack([ah, ax])
        y = np.array([0] * len(bh) + [1] * len(bx))
        return auc_sets(base, arc, y, folds, seed)

    res["E2_mage_raw"] = cond(bra, ara)
    res["E2_mage_para"] = cond(bpa, apa)

    res["E2_per_domain"] = {}
    for dom in pools["domains"]:
        hi = [i for i, d in enumerate(pools["dom_human"]) if d == dom]
        ri = [i for i, d in enumerate(pools["dom_ai_raw"]) if d == dom]
        pi = [i for i, d in enumerate(pools["dom_ai_para"]) if d == dom]
        if min(len(hi), len(ri), len(pi)) < 30:
            continue
        f = 5 if min(len(hi), len(ri), len(pi)) >= 50 else 3
        raw = auc_sets(np.vstack([bh[hi], bra[ri]]), np.vstack([ah[hi], ara[ri]]),
                       np.array([0] * len(hi) + [1] * len(ri)), f, seed)
        para = auc_sets(np.vstack([bh[hi], bpa[pi]]), np.vstack([ah[hi], apa[pi]]),
                        np.array([0] * len(hi) + [1] * len(pi)), f, seed)
        res["E2_per_domain"][dom] = {"n": [len(hi), len(ri), len(pi)], "raw": raw, "para": para}

    # E3 — ASAP quality
    import pandas as pd  # noqa: F401
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    df = load_asap(sets=(1, 2, 8), n_per_set=400, min_words=120, seed=seed)
    b, a = featurize(df["text"].tolist(), model, tok, dev, ac, 1000, "E3 asap")
    df["score_z"] = df.groupby("essay_set")["score"].transform(lambda x: (x - x.mean()) / (x.std() or 1.0))
    y = df["score_z"].values
    Xb = StandardScaler().fit_transform(np.column_stack([df["wl"].to_numpy(float), b]))  # length,mean,std,mad
    Xa = StandardScaler().fit_transform(a)
    kf = KFold(folds, shuffle=True, random_state=seed)
    r2b = float(cross_val_score(LinearRegression(), Xb, y, cv=kf, scoring="r2").mean())
    r2f = float(cross_val_score(LinearRegression(), np.hstack([Xb, Xa]), y, cv=kf, scoring="r2").mean())
    mb = sm.OLS(y, sm.add_constant(Xb)).fit()
    mf = sm.OLS(y, sm.add_constant(np.hstack([Xb, Xa]))).fit()
    F, p, _ = mf.compare_f_test(mb)
    res["E3_asap"] = {"cv_r2_baseline": round(r2b, 4), "cv_r2_arc": round(r2f, 4),
                      "cv_r2_delta": round(r2f - r2b, 4), "arc_F": round(float(F), 3),
                      "arc_p": float(p)}

    out = os.path.join(ROOT, "tables", f"results_{lm.replace('/', '_')}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("\n" + json.dumps({k: v for k, v in res.items() if k != "E2_per_domain"}, indent=2))
    print("per-domain:", json.dumps(res["E2_per_domain"], indent=2))
    print("-> ", out)


if __name__ == "__main__":
    main()
