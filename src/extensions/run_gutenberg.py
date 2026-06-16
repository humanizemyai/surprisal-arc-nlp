"""BOOK-LENGTH test: does an aesthetic surprisal arc exist in long literary prose?

The one regime the main study did not reach (essays maxed at ~626 words; Reagan's arcs were whole
novels). We fetch ~30 Project Gutenberg works, compute book-length surprisal via sliding windows,
and ask: (a) does the global/low-band arc structure grow with length (books vs essays vs QA)?
(b) do book arcs reduce to a few dominant SVD shapes (Reagan parallel)? Descriptive — no AI/label.

Run:  .venv/bin/python src/extensions/run_gutenberg.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.arc.features import band_energy_features, normalized_arc  # noqa: E402
from src.data.load_data import load_asap, load_hc3  # noqa: E402
from src.surprisal.extract import load_lm, token_surprisals, windowed_surprisals  # noqa: E402

N_BANDS = 8
ARC_LEN = 200
# Famous public-domain English literary works (Project Gutenberg ids)
BOOK_IDS = [1342, 84, 2701, 1661, 98, 1400, 11, 174, 345, 2542, 5200, 76, 1260, 768, 158,
            161, 219, 1232, 1184, 209, 120, 36, 35, 164, 2814, 730, 244, 1080, 768, 16, 1727, 100]
CACHE = os.path.join(ROOT, "data", "raw", "gutenberg")
UA = {"User-Agent": "Mozilla/5.0 (surprisal-arc research; contact hello@humanizemy.ai)"}


def fetch(bid: int) -> str | None:
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{bid}.txt")
    if os.path.exists(fp):
        return open(fp, encoding="utf-8", errors="ignore").read()
    for url in (f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
                f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
                f"https://www.gutenberg.org/ebooks/{bid}.txt.utf-8"):
        try:
            req = urllib.request.Request(url, headers=UA)
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            if len(raw) > 5000:
                open(fp, "w", encoding="utf-8").write(raw)
                time.sleep(0.4)
                return raw
        except Exception:
            continue
    return None


def strip_boiler(raw: str) -> str:
    s, e = raw.find("*** START OF"), raw.rfind("*** END OF")
    if s != -1:
        raw = raw[raw.find("\n", s) + 1:]
    if e != -1:
        raw = raw[:raw.rfind("*** END OF")] if "*** END OF" in raw else raw
    return raw.strip()


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "config.yaml")))
    lm = cfg["surprisal"]["models"][0]
    model, tok, dev = load_lm(lm, cfg["surprisal"]["device"])
    from tqdm import tqdm

    print(f"[1/4] Fetching {len(set(BOOK_IDS))} Gutenberg books + book-length surprisal...")
    book_bands, book_arcs, ok = [], [], 0
    for bid in tqdm(sorted(set(BOOK_IDS)), ncols=70):
        raw = fetch(bid)
        if not raw:
            continue
        txt = strip_boiler(raw)
        if len(txt) < 20000:
            continue
        txt = txt[int(len(txt) * 0.12):]                 # skip front matter / intro
        s = windowed_surprisals(txt, model, tok, dev, 1024, 512, 16384)
        if s.size < 2000:
            continue
        book_bands.append(band_energy_features(s, 128, N_BANDS))
        book_arcs.append(normalized_arc(s, ARC_LEN, 9))
        ok += 1
    print(f"      {ok} books processed")
    book_bands = np.array(book_bands); book_arcs = np.array(book_arcs)

    print("[2/4] Comparison: medium essays (ASAP) + short QA (HC3)...")
    ess = load_asap(sets=(1, 2, 8), n_per_set=40, min_words=120, seed=42)["text"].tolist()
    sht, _ = load_hc3("open_qa", 120, 40, 42)
    ess_bands, ess_arcs, sht_bands = [], [], []
    for t in tqdm(ess, ncols=70, desc="essays"):
        s = token_surprisals(t, model, tok, dev, 1000)
        ess_bands.append(band_energy_features(s, 128, N_BANDS)); ess_arcs.append(normalized_arc(s, ARC_LEN, 9))
    for t in tqdm(sht, ncols=70, desc="qa"):
        s = token_surprisals(t, model, tok, dev, 512)
        sht_bands.append(band_energy_features(s, 128, N_BANDS))
    ess_bands = np.array(ess_bands); ess_arcs = np.array(ess_arcs); sht_bands = np.array(sht_bands)

    print("[3/4] SVD of arcs (do a few shapes dominate?)...")
    def svd_var(arcs):
        A = arcs - arcs.mean(0)
        sv = np.linalg.svd(A, compute_uv=False)
        ev = sv ** 2 / (sv ** 2).sum()
        return ev, A

    book_ev, bookA = svd_var(book_arcs)
    ess_ev, _ = svd_var(ess_arcs)

    res = {
        "lm": lm, "n_books": ok,
        "low_band_share_band0": {"books": round(float(book_bands[:, 0].mean()), 4),
                                 "essays": round(float(ess_bands[:, 0].mean()), 4),
                                 "qa_short": round(float(sht_bands[:, 0].mean()), 4)},
        "low3_band_share": {"books": round(float(book_bands[:, :3].sum(1).mean()), 4),
                            "essays": round(float(ess_bands[:, :3].sum(1).mean()), 4),
                            "qa_short": round(float(sht_bands[:, :3].sum(1).mean()), 4)},
        "arc_svd_top_var": {"books_top1": round(float(book_ev[0]), 4),
                            "books_top2": round(float(book_ev[:2].sum()), 4),
                            "books_top3": round(float(book_ev[:3].sum()), 4),
                            "essays_top2": round(float(ess_ev[:2].sum()), 4)},
        "band_profile_books": [round(x, 4) for x in book_bands.mean(0)],
        "band_profile_essays": [round(x, 4) for x in ess_bands.mean(0)],
        "band_profile_qa": [round(x, 4) for x in sht_bands.mean(0)],
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(ROOT, "tables", "gutenberg_results.json"), "w"), indent=2)

    print("[4/4] Figure...")
    _fig(book_bands, ess_bands, sht_bands, book_arcs, book_ev, bookA, res)
    print("-> tables/gutenberg_results.json + figures/gutenberg_arc.png")


def _fig(bb, eb, sb, book_arcs, ev, bookA, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(N_BANDS)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(x, bb.mean(0), "-o", label=f"books (n={res['n_books']}, long)", color="#7C3AED")
    ax[0].plot(x, eb.mean(0), "-o", label="essays (ASAP)", color="#1B9E4B")
    ax[0].plot(x, sb.mean(0), "-o", label="QA (HC3, short)", color="#60A5FA")
    ax[0].set_title("Surprisal band-energy profile by text length")
    ax[0].set_xlabel("frequency band (low→high)"); ax[0].set_ylabel("relative power"); ax[0].legend()

    t = np.linspace(0, 1, ARC_LEN)
    for i in range(min(8, book_arcs.shape[0])):
        ax[1].plot(t, book_arcs[i], color="#bbb", lw=0.7)
    ax[1].plot(t, book_arcs.mean(0), color="#7C3AED", lw=2.5, label="mean book arc")
    ax[1].axhline(0, color="#888", lw=0.6, ls="--")
    ax[1].set_title("Book-length surprisal arcs (grey) + mean")
    ax[1].set_xlabel("normalized position in book"); ax[1].set_ylabel("z-surprisal"); ax[1].legend()

    k = min(8, len(ev))
    ax[2].bar(np.arange(k), ev[:k], color="#7C3AED")
    ax[2].set_title(f"Arc SVD scree (top-2 = {res['arc_svd_top_var']['books_top2']:.0%} of var)")
    ax[2].set_xlabel("component"); ax[2].set_ylabel("variance explained")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "gutenberg_arc.png"), dpi=140)


if __name__ == "__main__":
    main()
