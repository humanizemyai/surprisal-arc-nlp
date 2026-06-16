"""ARC-QUALITY ARM (closes the arc idea): does surprisal arc-SHAPE predict human-judged
writing quality among HUMANS? (different question from detection.)

ASAP-AES long sets (1/2/8). Per essay: surprisal (gpt2) -> mean, std, mad(=local uniformity/UID),
length, arc-shape(FFT). Target = within-set z-scored human holistic score. We ask whether arc-shape
adds predictive power for quality OVER length+mean+std+mad (nested F-test + 5-fold CV R²), and
whether high- vs low-quality essays have visibly different arcs.

Run:  .venv/bin/python src/stats/run_quality.py
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
from src.data.load_data import load_asap  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals  # noqa: E402

MAX_TOKENS = 1000  # set 8 runs long; gpt2 context = 1024


def main() -> None:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    ac = cfg["arc"]
    seed = cfg["detect"]["seed"]
    np.random.seed(seed)

    print("[1/4] Loading ASAP-AES long sets (1,2,8)...")
    df = load_asap(sets=(1, 2, 8), n_per_set=400, min_words=120, seed=seed)
    print(f"      n={len(df)} | per set {df['essay_set'].value_counts().sort_index().to_dict()}")

    model_name = cfg["surprisal"]["models"][0]
    print(f"[2/4] Loading LM '{model_name}' + extracting surprisal ({len(df)} essays)...")
    model, tok, dev = load_lm(model_name, cfg["surprisal"]["device"])
    print(f"      device={dev}")

    from tqdm import tqdm
    means, stds, mads, arcs, curves = [], [], [], [], []
    for t in tqdm(df["text"].tolist(), ncols=80):
        s = token_surprisals(t, model, tok, dev, MAX_TOKENS)
        b = baseline_features(s)
        means.append(b["mean"]); stds.append(b["std"]); mads.append(b["mad"])
        arcs.append(arc_shape_features(s, ac["resample_len"], ac["smooth_window"], ac["n_fft_coeffs"]))
        curves.append(normalized_arc(s, ac["resample_len"], ac["smooth_window"]))
    arcs = np.array(arcs); curves = np.array(curves)
    df["mean"], df["std"], df["mad"], df["length"] = means, stds, mads, df["wl"].astype(float)
    df["score_z"] = df.groupby("essay_set")["score"].transform(lambda x: (x - x.mean()) / (x.std() or 1.0))

    print("[3/4] Per-set Spearman (feature vs raw human score)...")
    from scipy.stats import spearmanr
    scal = ["length", "mean", "std", "mad"]
    spear = {}
    print(f"      {'set':>4} {'n':>5} " + " ".join(f"{f:>9}" for f in scal))
    for s, sub in df.groupby("essay_set"):
        rs = {f: float(spearmanr(sub[f], sub["score"]).statistic) for f in scal}
        spear[int(s)] = rs
        print(f"      {s:>4} {len(sub):>5} " + " ".join(f"{rs[f]:>+9.3f}" for f in scal))
    print("      (mad = local uniformity; r<0 => more-uniform essays score higher = UID-as-quality)")

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    Xb = StandardScaler().fit_transform(df[["length", "mean", "std", "mad"]].values)
    Xa = StandardScaler().fit_transform(arcs)
    y = df["score_z"].values

    import statsmodels.api as sm
    mb = sm.OLS(y, sm.add_constant(Xb)).fit()
    mf = sm.OLS(y, sm.add_constant(np.hstack([Xb, Xa]))).fit()
    F, p, ddf = mf.compare_f_test(mb)

    kf = KFold(5, shuffle=True, random_state=seed)
    r2b = float(cross_val_score(LinearRegression(), Xb, y, cv=kf, scoring="r2").mean())
    r2f = float(cross_val_score(LinearRegression(), np.hstack([Xb, Xa]), y, cv=kf, scoring="r2").mean())

    print("\n[4/4] Pooled (within-set z-scored quality):")
    print(f"      baseline in-sample R²={mb.rsquared:.3f} | +arc R²={mf.rsquared:.3f}")
    print(f"      arc block nested F={F:.2f}  p={p:.2e}  (df_num={ddf:.0f})")
    print(f"      CV-R²: baseline={r2b:.3f}  +arc={r2f:.3f}  delta={r2f - r2b:+.3f}")
    arc_helps = (p < 0.05) and ((r2f - r2b) >= 0.01)
    print(f"\n[VERDICT] arc predicts quality? {'YES' if arc_helps else 'NO'}  -> "
          f"{'aesthetic-arc lives in the quality domain' if arc_helps else 'arc fully closed (dead for detection AND quality)'}")

    out = {"dataset": "ASAP-AES sets 1/2/8", "lm": model_name, "n": int(len(df)),
           "per_set_spearman_vs_score": spear,
           "pooled": {"baseline_R2_insample": round(mb.rsquared, 4),
                      "full_R2_insample": round(mf.rsquared, 4),
                      "arc_block_F": round(float(F), 3), "arc_block_p": float(p),
                      "cv_r2_baseline": round(r2b, 4), "cv_r2_full": round(r2f, 4),
                      "cv_r2_delta": round(r2f - r2b, 4)},
           "verdict_arc_predicts_quality": bool(arc_helps)}
    json.dump(out, open(os.path.join(ROOT, "tables", "quality_results.json"), "w"), indent=2)

    _save_figure(curves, df, r2b, r2f, model_name)
    print("      -> tables/quality_results.json + figures/quality_arc.png")


def _save_figure(curves, df, r2b, r2f, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = df.groupby("essay_set")["score"].transform(lambda x: x >= x.median()).to_numpy().astype(bool)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.linspace(0, 1, curves.shape[1])
    ax[0].plot(x, curves[hi].mean(0), label=f"high-score (n={int(hi.sum())})", color="#1B9E4B", lw=2)
    ax[0].plot(x, curves[~hi].mean(0), label=f"low-score (n={int((~hi).sum())})", color="#EC4899", lw=2)
    ax[0].axhline(0, color="#888", lw=0.6, ls="--")
    ax[0].set_title("Mean normalized surprisal arc — ASAP (high vs low quality)")
    ax[0].set_xlabel("normalized position"); ax[0].set_ylabel("z-surprisal"); ax[0].legend()

    ax[1].bar(["baseline\n(len+mean+std+mad)", "+ arc shape"], [r2b, r2f], color=["#9CA3AF", "#A855F7"])
    ax[1].set_title(f"5-fold CV R² predicting human quality (LM={model_name})")
    ax[1].set_ylabel("CV R²"); ax[1].axhline(0, color="#888", lw=0.6)
    for i, v in enumerate([r2b, r2f]):
        ax[1].text(i, v + 0.005, f"{v:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "quality_arc.png"), dpi=140)


if __name__ == "__main__":
    main()
