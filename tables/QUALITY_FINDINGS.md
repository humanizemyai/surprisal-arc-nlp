# Arc-quality findings — ASAP-AES (long sets 1/2/8, gpt2 surprisal, n=1200)

## Per-set Spearman (feature vs raw human score)
| set | n   | length | mean(ppl) | std   | mad   |
|-----|-----|--------|-----------|-------|-------|
| 1   | 400 | +0.748 | -0.193    | -0.086| +0.046|
| 2   | 400 | +0.646 | -0.292    | -0.188| +0.075|
| 8   | 400 | +0.460 | -0.205    | -0.032| +0.115|

## Pooled (within-set z-scored quality)
- baseline (length+mean+std+mad) in-sample R² = 0.352; **+arc R² = 0.359**
- arc block nested F = 1.79, **p = 0.076 (n.s.)**
- **CV-R²: baseline 0.341 vs +arc 0.340 (delta -0.001)** — arc adds nothing out-of-sample
- Figure `figures/quality_arc.png`: high- vs low-score mean arcs overlap.

## Verdict: arc predicts quality? NO — the arc is now FULLY closed (dead for detection AND quality).

## Interpretation
- **Length dominates quality** (Spearman +0.46 to +0.75) — the well-known AES baseline.
- Among surprisal features, **lower perplexity and lower variance (more uniform) mildly predict higher
  quality** (mean r<0 in all sets; std r<0 in sets 1/2). This is the **UID-consistent** direction and the
  **opposite** of the aesthetic-arc prediction (partly length-entangled; pooled control still leaves arc null).
- The **aesthetic surprisal arc does not exist** — neither as a human/AI tell nor as a quality signal.

## Bottom line for the project
The music->language "aesthetic arc" transfer is falsified across three external benchmarks. What remains
true and useful: **UID-aligned statistics (level + uniformity), not arc shape, carry the signal** — and
they behave differently under paraphrase attacks (see DECIDER_FINDINGS.md).
