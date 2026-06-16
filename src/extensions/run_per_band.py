"""PER-BAND extension: is the signal in a MID frequency band we collapsed?

The main study used only the LOW band (arc shape) + a scalar variance (burstiness). This tests the
full multi-band surprisal "EQ" (8 equal frequency bands, relative power) — the DJ-EQ / E(tau) idea —
on (a) MAGE detection (raw + paraphrased) and (b) ASAP quality (+ longest set 8). If a band-energy
vector adds AUC/R² over mean+burstiness, the structural-surprisal hypothesis survives in a band we
missed; if not, the negative is even stronger.

Run:  ARC_LM=gpt2 .venv/bin/python src/extensions/run_per_band.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.arc.features import arc_shape_features, band_energy_features, baseline_features  # noqa: E402
from src.data.load_data import load_asap, load_mage_ood  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals  # noqa: E402

N_BANDS = 8


def feats(texts, model, tok, dev, ac, maxtok, desc=""):
    from tqdm import tqdm
    base, arc, band = [], [], []
    for t in tqdm(texts, ncols=70, desc=desc):
        s = token_surprisals(t, model, tok, dev, maxtok)
        b = baseline_features(s)
        base.append([b["mean"], b["std"], b["mad"]])
        arc.append(arc_shape_features(s, ac["resample_len"], ac["smooth_window"], ac["n_fft_coeffs"]))
        band.append(band_energy_features(s, 128, N_BANDS))
    return np.array(base), np.array(arc), np.array(band)


def cvauc(X, y, folds, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return round(float(cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean()), 3)


def cvr2(X, y, folds, seed):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    kf = KFold(folds, shuffle=True, random_state=seed)
    return round(float(cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()), 4)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    ac, seed, folds = cfg["arc"], cfg["detect"]["seed"], cfg["detect"]["cv_folds"]
    lm = os.environ.get("ARC_LM", cfg["surprisal"]["models"][0])
    np.random.seed(seed)
    model, tok, dev = load_lm(lm, cfg["surprisal"]["device"])
    out = {"lm": lm, "n_bands": N_BANDS}

    # ---- MAGE detection ----
    pools = load_mage_ood(min_words=60, max_per_cell=300, seed=seed)
    bh, arh, bnh = feats(pools["human"], model, tok, dev, ac, 768, "MAGE human")
    bra, arr, bnr = feats(pools["ai_raw"], model, tok, dev, ac, 768, "MAGE raw")
    bpa, arp, bnp = feats(pools["ai_para"], model, tok, dev, ac, 768, "MAGE para")

    def detect(bx, arx, bnx):
        MB = np.vstack([bh, bx])
        ARC = np.vstack([arh, arx])
        BND = np.vstack([bnh, bnx])
        y = np.array([0] * len(bh) + [1] * len(bx))
        return {"mean+burst": cvauc(MB, y, folds, seed),
                "+arc": cvauc(np.hstack([MB, ARC]), y, folds, seed),
                "+band": cvauc(np.hstack([MB, BND]), y, folds, seed),
                "band_only": cvauc(BND, y, folds, seed)}

    out["MAGE_raw"] = detect(bra, arr, bnr)
    out["MAGE_para"] = detect(bpa, arp, bnp)
    out["MAGE_band_profile"] = {"human": [round(x, 4) for x in bnh.mean(0)],
                                "ai_raw": [round(x, 4) for x in bnr.mean(0)],
                                "ai_para": [round(x, 4) for x in bnp.mean(0)]}

    # ---- ASAP quality ----
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm
    df = load_asap(sets=(1, 2, 8), n_per_set=400, min_words=120, seed=seed)
    bq, arq, bnq = feats(df["text"].tolist(), model, tok, dev, ac, 1000, "ASAP")
    df["score_z"] = df.groupby("essay_set")["score"].transform(lambda x: (x - x.mean()) / (x.std() or 1.0))

    def quality(mask, tag):
        y = df["score_z"].to_numpy()[mask]
        MB = StandardScaler().fit_transform(np.column_stack([df["wl"].to_numpy(float)[mask], bq[mask]]))
        BND = StandardScaler().fit_transform(bnq[mask])
        ARC = StandardScaler().fit_transform(arq[mask])
        r2_mb = cvr2(MB, y, folds, seed)
        r2_band = cvr2(np.hstack([MB, BND]), y, folds, seed)
        r2_arc = cvr2(np.hstack([MB, ARC]), y, folds, seed)
        mb = sm.OLS(y, sm.add_constant(MB)).fit()
        mf = sm.OLS(y, sm.add_constant(np.hstack([MB, BND]))).fit()
        F, p, _ = mf.compare_f_test(mb)
        return {"n": int(mask.sum()), "cv_r2_baseline": r2_mb, "cv_r2_+band": r2_band,
                "cv_r2_+arc": r2_arc, "band_delta": round(r2_band - r2_mb, 4),
                "band_F": round(float(F), 3), "band_p": float(p)}

    allmask = np.ones(len(df), dtype=bool)
    s8mask = (df["essay_set"] == 8).to_numpy()
    out["ASAP_all"] = quality(allmask, "all")
    out["ASAP_set8_longest"] = quality(s8mask, "set8")
    hi = df.groupby("essay_set")["score"].transform(lambda x: x >= x.median()).to_numpy().astype(bool)
    out["ASAP_band_profile"] = {"high": [round(x, 4) for x in bnq[hi].mean(0)],
                                "low": [round(x, 4) for x in bnq[~hi].mean(0)]}

    json.dump(out, open(os.path.join(ROOT, "tables", "perband_results.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if "profile" not in k}, indent=2))
    _fig(bnh, bnr, bnp, bnq, hi, out)
    print("-> tables/perband_results.json + figures/per_band.png")


def _fig(bnh, bnr, bnp, bnq, hi, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(N_BANDS)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(x, bnh.mean(0), "-o", label="human", color="#1B9E4B")
    ax[0].plot(x, bnr.mean(0), "-o", label="GPT4 raw", color="#60A5FA")
    ax[0].plot(x, bnp.mean(0), "-o", label="GPT4 para", color="#EC4899")
    ax[0].set_title("MAGE: surprisal band-energy profile")
    ax[0].set_xlabel("frequency band (low→high)"); ax[0].set_ylabel("relative power"); ax[0].legend()
    ax[1].plot(x, bnq[hi].mean(0), "-o", label="high-score", color="#1B9E4B")
    ax[1].plot(x, bnq[~hi].mean(0), "-o", label="low-score", color="#EC4899")
    ax[1].set_title("ASAP: band-energy profile (quality)")
    ax[1].set_xlabel("frequency band (low→high)"); ax[1].set_ylabel("relative power"); ax[1].legend()
    labels = ["mean+burst", "+arc", "+band"]
    raw = [out["MAGE_raw"][k] if k != "+band" else out["MAGE_raw"]["+band"] for k in labels]
    para = [out["MAGE_para"][k] for k in labels]
    xi = np.arange(3); w = 0.38
    ax[2].bar(xi - w / 2, raw, w, label="MAGE raw", color="#A855F7")
    ax[2].bar(xi + w / 2, para, w, label="MAGE para", color="#EC4899")
    ax[2].axhline(0.5, color="#888", ls="--", lw=0.8)
    ax[2].set_xticks(xi); ax[2].set_xticklabels(labels); ax[2].set_ylim(0.4, 1.0)
    ax[2].set_title("Does +band beat mean+burst? (AUC)"); ax[2].set_ylabel("ROC-AUC"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "per_band.png"), dpi=140)


if __name__ == "__main__":
    main()
