# There Is No Aesthetic Surprisal Arc: What Token Surprisal Does and Doesn't Reveal About Machine-Generated Text and Writing Quality

**Berk Ustun**¹

¹ Independent researcher (computational linguistics). Correspondence: hello@humanizemy.ai

*Preprint — June 2026. Prepared for public dissemination via ResearchGate.*

---

## Abstract

A productive analogy holds that great music shapes a listener's surprise into a deliberate arc — tension that rises to a climax and resolves. We ask whether written language carries the same structure: does an author shape *information-theoretic surprisal* into a global arc, and if so, is that arc (i) a signature that separates human from machine-generated text, or (ii) a correlate of human-judged writing quality? Using only public benchmarks — **HC3**, **MAGE** (including a paraphrase attack) and **ASAP-AES** — and per-token surprisal from open language models, we operationalize the arc as the low-frequency spectral shape of the length-normalized surprisal sequence, with the surprisal *level* (mean / perplexity) and *variance* (burstiness) as baselines. Across three experiments we find a clean negative: the arc shape is **at chance for human/AI detection** (ROC-AUC 0.55 on HC3, 0.51–0.53 on long-form MAGE) and **adds nothing to writing-quality prediction** (ΔCV-R² = −0.001, nested-*F* p = 0.08 on ASAP-AES). What carries signal is mundane: the *level* and *variance* of surprisal. Raw machine text is markedly more uniform and predictable than human text — it **overshoots Uniform Information Density** — and among human essays, more uniform, lower-perplexity writing is rated *higher*, not lower, the opposite of the aesthetic-arc prediction. Finally, the two statistics behave differently under attack: paraphrasing **collapses mean perplexity** (AUC 0.80 → 0.58) while **leaving burstiness intact** (0.80 → 0.78), making burstiness the attack-robust statistic. All results replicate with a second, larger surprisal model (Pythia-410m): the arc stays at chance, mean perplexity again collapses under paraphrase (0.83 → 0.55) while burstiness holds (0.83 → 0.77), and the arc again fails to predict quality (nested-*F* p = 0.31). A fuller multi-band ("EQ") view of the whole surprisal spectrum recovers only a statistically-detectable but practically-negligible sliver of structure (quality nested-*F* p < 0.01 yet ΔCV-R² ≈ +0.005; detection ≤ +0.01 AUC over mean+burstiness), so the negative concerns surprisal *structure* broadly, not just the low-frequency operationalization. Even **book-length literary prose** (30 Project Gutenberg works) shows no arc — surprisal there is the *flattest, most broadband* of all text types, and book arcs do not reduce to a few shapes (unlike emotional arcs). We release all code and a fully reproducible pipeline.

**Keywords:** surprisal · uniform information density · machine-generated text detection · burstiness · writing quality · psycholinguistics · negative result

---

## 1. Introduction

In a recurring analogy between music and language, beauty is described as *managed surprise*: a great composition is neither monotonous (fully predictable) nor noise (fully unpredictable), but rides an inverted-U "sweet spot," shaping expectation into a deliberate arc that builds tension and resolves it (Meyer 1956; Huron 2006; Wundt's inverted-U; Birkhoff 1933). If language works the same way, an author's *information-theoretic surprisal* — how unexpected each token is given its context — should trace a comparable arc over a text, and that arc might be both a fingerprint of human authorship and a correlate of quality.

This paper tests that hypothesis directly. We treat the per-token surprisal sequence of a text as a signal, smooth it into a latent curve T(t), and isolate its **shape** (independent of overall level and variance) as the low-frequency content of the normalized curve. We then ask three concrete questions on public data:

1. **Detection.** Does the surprisal-arc *shape* separate human from machine-generated text?
2. **Robustness.** Does any such signal survive a paraphrase (humanization) attack that is designed to evade detectors?
3. **Quality.** Among human texts, does the arc *shape* predict human-judged writing quality?

Our answer to all three is negative for the *shape*, and the negative is informative. There is a real and long-standing tension in psycholinguistics between **Uniform Information Density** (UID; Levy & Jaeger 2007; Jaeger 2010), which says efficient producers *flatten* surprisal, and the aesthetic view, which says skilled producers *shape* it. Our data adjudicate this: the *shape* carries no signal, while UID-aligned statistics (level and uniformity) do — and, for quality, in the UID-consistent direction. We also surface a useful by-product for detection research: under a paraphrase attack, the surprisal *level* and *variance* are not equally fragile.

We deliberately use only **independent, public, multi-author benchmarks** and report a falsifiable negative with explicit controls (text length, an adversarial regime, per-domain stratification, and a second surprisal model), in the spirit of honest, reproducible evaluation.

## 2. Related work

**Surprisal in language.** Surprisal theory (Hale 2001; Levy 2008) links a word's processing cost to its negative log-probability under context. Building on it, the Uniform Information Density hypothesis (Levy & Jaeger 2007; Jaeger 2010) holds that producers distribute information evenly to ease processing — i.e., they *minimize* surprisal variance. This predicts that natural, fluent text should be *flat*, not arc-shaped.

**Aesthetics as managed surprise.** A parallel tradition models aesthetic value as optimal surprise: Birkhoff's (1933) order/complexity ratio, Meyer's (1956) expectation–tension–resolution, Huron's (2006) ITPRA, and the empirical "six basic shapes" of narrative emotional arcs recovered by Reagan et al. (2016). Transferred to language, this predicts a deliberately *shaped*, non-flat information curve. UID and the aesthetic view therefore make opposite predictions about the surprisal trajectory — the tension this paper tests.

**Machine-text detection.** Zero-shot detectors lean on exactly these statistics: mean log-probability / perplexity and its variation ("burstiness"), e.g. GPTZero, DetectGPT (Mitchell et al. 2023) and Binoculars (Hans et al. 2024). Shared benchmarks — RAID (Dugan et al. 2024), MAGE (Li et al. 2024), HC3 (Guo et al. 2023) — show detectors are brittle under adversarial paraphrasing and humanization. We ask whether arc *shape* offers a signal these scalar statistics miss, and whether it is more attack-robust.

## 3. Data

All data are public and multi-author; no in-house corpus is used.

- **HC3** (Guo et al. 2023) — human vs ChatGPT answers (`open_qa`). Short (~100 words). Used for the easy-regime detection test (E1).
- **MAGE** (Li et al. 2024) — multi-domain human vs machine text "in the wild." We use the GPT-4 out-of-distribution split (domains: CNN news, DialogSum, IMDB, PubMed) in two conditions: raw GPT-4 and its **paraphrased** version (the adversarial / humanization attack). Long-form (median ~205 words). Used for E2.
- **ASAP-AES** (Hewlett Foundation) — student essays with independent **human holistic scores**. We use the long sets 1, 2 and 8 (median 365 / 368 / 626 words; argumentative and narrative). Used for the quality test (E3).

## 4. Methods

**Surprisal.** For each text we compute per-token surprisal sᵢ = −log₂ p(xᵢ | x_<ᵢ) with an open causal language model (primary: GPT-2; robustness: Pythia-410m), via Hugging Face `transformers`.

**Arc shape vs. baselines.** From each surprisal sequence we derive:
- *mean* — average surprisal (a perplexity proxy; the surprisal **level**);
- *burstiness* — its standard deviation and mean absolute adjacent difference (the surprisal **variance** / local roughness; `mad` doubles as a *local-uniformity* / UID statistic);
- *arc shape* — we resample the sequence to a fixed length (64), low-pass smooth it to a curve T(t), **z-normalize it (removing mean and variance)**, and take the magnitudes of its first eight non-DC FFT coefficients. By construction this captures the global **shape** only, independent of level and variance — exactly the aesthetic-arc claim.

**Detection (E1, E2).** Logistic regression with 5-fold stratified cross-validation; we report ROC-AUC for each feature family (mean, burstiness, arc-shape, mean+burstiness). For E2 we evaluate both the raw and paraphrased conditions and **stratify per domain** to rule out composition artifacts.

**Quality (E3).** Target = human holistic score, z-scored *within* essay set (scales differ per set). We regress it on a baseline block (length, mean, std, mad) and test whether the arc-shape block adds predictive power via (a) a nested *F*-test and (b) the change in 5-fold cross-validated R². We also report per-set Spearman correlations of each scalar feature with the raw score.

**Robustness.** Every experiment is repeated with a second, larger surprisal model (Pythia-410m) to confirm findings are not model-specific.

## 5. Results

### 5.1 E1 — Detection, easy regime (HC3)
On raw human-vs-ChatGPT text, the surprisal **level and variance separate the classes almost perfectly**, while **arc shape is at chance**: ROC-AUC mean = **0.969**, burstiness = **0.953**, **arc-shape = 0.552**, mean+burstiness = 0.967 (Figure 1). The normalized mean arcs of human and AI text are visually indistinguishable. Adding the arc to the baseline does not improve it.

![Figure 1](figures/pilot_arc_vs_burstiness.png)
*Figure 1 — E1: mean normalized surprisal arc (human vs AI) and per-family ROC-AUC.*

### 5.2 E2 — Detection, long-form and under attack (MAGE)
On long-form text the picture is the same for the arc and newly informative for the baselines (Figure 2):

| condition | mean (ppl) | burstiness | arc-shape | mean+burst |
|---|---|---|---|---|
| raw (human vs GPT-4)            | 0.804 | 0.804 | 0.520 | 0.842 |
| **paraphrased** (human vs GPT-4-para) | **0.580** | **0.781** | 0.517 | 0.845 |

Two results. (i) **Arc shape is again at chance** (0.52 / 0.52), even in the long-form, adversarial regime where the hypothesis was most likely to help. (ii) The paraphrase attack is **selective**: it collapses the discriminative power of **mean perplexity** (0.80 → 0.58, toward chance) but leaves **burstiness** largely intact (0.80 → 0.78), so the combined detector is unharmed (0.84 → 0.85). **Burstiness is the attack-robust surprisal statistic.** Per-domain stratification (CNN, DialogSum, IMDB, PubMed) reproduces both results in every domain: arc-shape stays at 0.43–0.55 throughout, and mean perplexity drops from raw to paraphrased in all four domains (e.g. IMDB 0.97 → 0.84, PubMed 0.93 → 0.80) — the pooled finding is not a composition artifact.

![Figure 2](figures/decider_arc_paraphrase.png)
*Figure 2 — E2: mean arcs (human / GPT-4 / GPT-4-paraphrased) and raw-vs-paraphrased ROC-AUC by feature family.*

### 5.3 E3 — Writing quality (ASAP-AES)
Among human essays, **length dominates** quality (per-set Spearman ρ = +0.75 / +0.65 / +0.46 for sets 1 / 2 / 8). Among surprisal features, **lower perplexity and lower variance (more uniform) mildly predict higher scores** (mean ρ ≈ −0.19 to −0.29; std ρ ≈ −0.09 to −0.19 in sets 1–2) — the **UID-consistent** direction and the *opposite* of the aesthetic-arc prediction. The **arc shape adds nothing**: over a length+mean+std+mad baseline, the arc block is not significant (nested *F* = 1.79, p = 0.076) and *worsens* out-of-sample fit (5-fold CV-R² 0.341 → 0.340, Δ = −0.001). High- and low-scoring essays have overlapping mean arcs (Figure 3).

![Figure 3](figures/quality_arc.png)
*Figure 3 — E3: mean normalized arc of high- vs low-scoring essays, and CV-R² for quality with and without the arc block.*

### 5.4 Robustness (second surprisal model)
Repeating all three experiments with **Pythia-410m** (≈3.3× larger than GPT-2) reproduces every qualitative result, confirming they are not artifacts of one surprisal model. **E1:** arc-shape 0.53 (vs mean 0.96, burstiness 0.97). **E2:** arc-shape 0.53 raw / 0.53 paraphrased; mean perplexity again **collapses 0.83 → 0.55** under the paraphrase attack while **burstiness holds 0.83 → 0.77**; per-domain, arc-shape stays at chance in every domain (the lone exception — PubMed raw, 0.67 — is small-*n*, does not appear under GPT-2, and disappears under paraphrase). **E3:** the arc block again fails to predict quality (ΔCV-R² = −0.003; nested-*F* = 1.18, p = 0.31). The negative is thus **model-robust, length-robust, regime-robust (raw + adversarial) and domain-robust**; the perplexity-fragility / burstiness-robustness contrast under attack is likewise reproduced. (Per-LM numbers: `tables/results_gpt2.json`, `tables/results_EleutherAI_pythia-410m.json`.)

![Figure 4](figures/robustness_two_lms.png)
*Figure 4 — Robustness across two surprisal models: arc-shape stays at chance, and the perplexity-collapse / burstiness-hold under paraphrase reproduces.*

### 5.5 A fuller multi-band view (per-band "EQ")
The arc-shape feature used only the *low-frequency* band. To check whether a *mid* band carries signal the single arc missed, we split the whole surprisal spectrum into eight equal frequency bands (relative power per band — the DJ-EQ / E(τ) view). It helps — barely. The band-energy distribution is weakly discriminative on its own (human-vs-AI AUC 0.59 raw / 0.62 paraphrased, above the arc's 0.52) and, for quality, the band block is statistically significant (nested-*F* p = 0.008 over all essays; p = 0.014 on the longest set 8). But it adds essentially nothing over the level+variance baselines: **+0.011 AUC** over mean+burstiness on raw MAGE, **≈ 0** under paraphrase, and **+0.005 CV-R²** for quality — which turns *negative* (−0.003) on the longest essays. Higher-quality essays do spread surprisal slightly more evenly across bands (less low-band dominance; Figure 5), consistent with the UID reading. The refinement: it is the low-frequency *arc* specifically that is null, while the broader spectrum holds a faint, practically-unusable structural signal.

*Figure 5 — Per-band surprisal-energy profiles (MAGE human / GPT-4 / paraphrased; ASAP high- vs low-quality) and incremental detection AUC.* (`per_band.png`)

### 5.6 Book-length literary prose (Gutenberg)
The arc's best hope was long-form: Reagan et al. (2016) recovered six dominant *emotional* arc shapes from whole novels. We computed book-length *surprisal* arcs for 30 public-domain literary works (windowed GPT-2 over ~16k-token passages) and find the opposite of the hypothesis. (i) Low-frequency (global-arc) energy **decreases** with length: the lowest band holds 0.30 of the spectrum in short QA, 0.15 in essays, but only **0.12 in books** — book-length surprisal is the *flattest, most broadband* of the three, the least arc-dominated; the only strongly low-frequency component is the trivial high-surprisal *opening transient* of short texts, not a tension curve. (ii) Book surprisal arcs do **not** reduce to a few shapes — the top two SVD components explain just **22%** of variance (Figure 6), unlike Reagan's emotional arcs. **Surprisal arcs are not emotional arcs.** The negative is therefore complete across scales — QA, essays, and whole books.

*Figure 6 — Book-length surprisal: band-energy by text length (books flattest), 30 book arcs with their near-flat mean, and a gradual SVD scree (no dominant shapes).* (`gutenberg_arc.png`)

## 6. Discussion

**There is no aesthetic surprisal arc.** Across an easy detection regime, a long-form adversarial regime, and a human-quality regression, the *shape* of the surprisal trajectory carries no signal. The music→language transfer of "managed surprise" — at least when operationalized as the spectral shape of token surprisal — does not hold.

**UID beats the aesthetic view for these phenomena.** What separates human from machine text, and what tracks human quality, is the *level* and *uniformity* of surprisal, not its shape — and for quality the direction is UID-consistent (more uniform, lower perplexity → rated higher). Machine text is *too* uniform and predictable: it **overshoots** UID relative to humans, which is precisely why simple perplexity/burstiness detectors work on raw model output.

**A practical note for detection.** The surprisal *level* and *variance* are not equally robust to humanization. A paraphrase attack restores a human-like perplexity *level* but does not restore human-like *burstiness*, leaving variance-based features as the more durable signal. Detectors (and, symmetrically, anyone studying humanization) should weight burstiness accordingly rather than relying on perplexity level.

## 7. Limitations

Surprisal is computed with modest open models (GPT-2, Pythia-410m), not the generators themselves; a larger surrogate could shift absolute AUCs (though §5.4 shows the qualitative pattern is stable). "Arc shape" is one reasonable operationalization (low-frequency FFT of a 64-point normalized curve); we also tested the fuller multi-band spectrum (§5.5) and book-length literary prose (§5.6), both negative, but we cannot prove *no* shape-like feature ever helps (a sentence-level arc or a wavelet basis could still differ) — only that these natural families do not. MAGE's paraphrase is one attack; RAID's broader attack suite would sharpen the robustness map. ASAP essays are a student-writing genre; literary corpora might host arcs that classroom essays do not.

## 8. Conclusion

Tested honestly on public benchmarks, the appealing idea that writers shape surprisal into an aesthetic arc does not survive contact with data: the arc shape predicts neither authorship nor quality. The signal lives in the boring statistics — the level and uniformity of surprisal — consistent with Uniform Information Density rather than managed-surprise aesthetics, with machine text distinguished by *over-*uniformity and burstiness proving the more attack-robust tell. We release the full pipeline so the negative can be checked and extended.

## Reproducibility

Code, configuration and result tables: `github.com/<repo>`. All datasets are public (HC3, MAGE, ASAP-AES). `src/run_all.py` reproduces every number with `ARC_LM=gpt2` and `ARC_LM=EleutherAI/pythia-410m`.

## References (to finalize)

Birkhoff (1933); Dugan et al. (2024, RAID); Guo et al. (2023, HC3); Hale (2001); Hans et al. (2024, Binoculars); Huron (2006); Jaeger (2010); Levy (2008); Levy & Jaeger (2007); Li et al. (2024, MAGE); Meyer (1956); Mitchell et al. (2023, DetectGPT); Reagan et al. (2016).
