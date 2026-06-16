# Project summary — surprisal, the (absent) aesthetic arc, and what actually carries signal

Three experiments on **external, independent benchmarks** (gpt2 surprisal, 5-fold).

## E1 — Detection, easy regime (HC3, human vs ChatGPT, ~100 words)
ROC-AUC: mean-perplexity **0.97**, burstiness **0.95**, **arc-shape 0.55 (chance)**, all 0.97.
=> Level + uniformity separate raw AI almost perfectly; arc-shape carries nothing.

## E2 — Detection, hard/adversarial regime (MAGE GPT4, long-form ~205 words, raw vs paraphrased)
| | mean-ppl | burstiness | arc-shape | mean+burst |
|---|---|---|---|---|
| raw  | 0.80 | 0.81 | 0.51 | 0.85 |
| paraphrased | **0.60** | **0.79** | 0.53 | 0.85 |
=> Paraphrase **collapses perplexity** but **not burstiness**; **arc-shape still chance**.
**Burstiness is the attack-robust surprisal statistic.**

## E3 — Quality (ASAP-AES long sets 1/2/8, human holistic scores, n=1200)
Length dominates (Spearman +0.46..+0.75). Lower perplexity + more uniformity mildly predict higher
quality (UID direction). **Arc-shape adds nothing** (nested F p=0.076 n.s.; CV-R² delta -0.001).

## Thesis (one coherent paper, external data only)
1. **There is no aesthetic surprisal arc.** The music->language transfer is falsified for detection
   (E1, E2 incl. a paraphrase attack that demonstrably collapses perplexity) and for quality (E3).
   Well-controlled negative (length, regime, attack, two-tertile descriptive overlap).
2. **UID beats aesthetic shaping for quality.** Uniformity/low-perplexity (UID-aligned), not a shaped
   arc, track human quality (alongside length).
3. **AI overshoots UID, and attacks are selective.** AI text is too uniform/predictable (detectable);
   paraphrase restores the *level* (perplexity) but leaves the *variance* (burstiness) detectable.

## Robustness (DONE — confirms all findings)
- **2nd surprisal LM (Pythia-410m):** every result replicates. E1 arc 0.53; E2 arc 0.53/0.53, mean
  0.83->0.55 under paraphrase, burst 0.83->0.77; E3 arc n.s. (F p=0.31, dCV-R2 -0.003).
  -> null is **model-robust**. (`results_gpt2.json` vs `results_EleutherAI_pythia-410m.json`)
- **Per-domain MAGE (cnn/dialogsum/imdb/pubmed):** arc at chance in every domain (both LMs); mean drops
  raw->para in every domain. -> pooled finding is **not a composition artifact**.
- Figure: `figures/robustness_two_lms.png`.
- Still optional (not required): RAID's 11 attacks for a fuller attack x statistic map.

## Per-band "EQ" extension (DONE — addresses per-token vs per-band critique)
Tested the full surprisal spectrum as 8 frequency bands (relative power), not just the low-band arc.
- Detection: band-only AUC 0.59 raw / 0.62 para (> arc 0.52, < mean+burst); +band over mean+burst = +0.011 raw, ~0 para.
- Quality: band block significant (nested-F p=0.008 all; p=0.014 set8) but dCV-R2 only +0.005 (all), -0.003 (set8 longest). High-quality essays spread surprisal slightly more evenly (less low-band dominance).
- Verdict: the per-band view DOES see a bit more than the single arc (the critique was directionally right), but it's a statistically-detectable, practically-negligible sliver — doesn't change the headline. `tables/perband_results.json`, `figures/per_band.png`.
## Book-length literary prose (DONE — closes the 'long human text' gap)
30 Project Gutenberg works, windowed GPT-2 over ~16k-token passages.
- Low-band (global-arc) share DECREASES with length: QA 0.30 -> essays 0.15 -> books 0.12. Books have the
  FLATTEST, most broadband surprisal spectrum (least arc-structured). The only strong low-freq component is
  the trivial opening transient of short text.
- Book arcs do NOT reduce to a few shapes: SVD top-2 = 22% of variance (vs Reagan's six emotional-arc shapes).
- => Surprisal arcs are NOT emotional arcs. The negative is complete across scales (QA -> essays -> novels).
- `tables/gutenberg_results.json`, `figures/gutenberg_arc.png`.
