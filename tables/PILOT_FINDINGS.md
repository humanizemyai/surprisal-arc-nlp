# Pilot findings — HC3 + gpt2 surprisal

## Result (5-fold ROC-AUC; 100 human + 100 ChatGPT; HC3/open_qa; LM = gpt2)
| feature set            | AUC            |
|------------------------|----------------|
| mean perplexity        | 0.969 ± 0.021  |
| burstiness (std + mad) | 0.953 ± 0.030  |
| **arc shape (FFT)**    | **0.552 ± 0.100** |
| all                    | 0.969 ± 0.029  |

Gate (arc-shape ≥ 0.65 **and** > burstiness + 0.03): **FAIL.**
Figure: `figures/pilot_arc_vs_burstiness.png` — human & AI mean normalized arcs are nearly identical.

## Honest interpretation
- In the **easy** regime (raw human vs ChatGPT, short QA) surprisal **level** (mean = perplexity)
  and **variance** (burstiness) already separate the classes almost perfectly. AI text is
  lower-perplexity and more uniform → consistent with AI **overshooting Uniform Information
  Density** (too predictable, too flat).
- The **normalized arc shape** carries essentially no independent signal (≈chance) and adds
  nothing over the mean. Human and AI share one generic curve: high-surprisal opening → decay
  as context accrues → end artifact. That is information-decay, not narrative tension.

## Diagnosis (suspect our own setup first)
This pilot likely tested the wrong conditions for the arc hypothesis:
1. **Texts too short** (~100 words). A global arc (rise → climax → cadence) is a long-form
   property (Reagan's story arcs were whole novels). ~100-word QA cannot express one.
2. **Easy regime.** Raw AI is trivially separable by level → no headroom. The arc hypothesis was
   motivated by the **hard** regime where level/variance FAIL (in-house humanized-text:
   burstiness AUC ≈ 0.42).

## Decider experiment (next)
Long-form + adversarial human-vs-AI (RAID adversarial subset; long essays). Test whether
arc-shape adds incremental AUC **where mean + burstiness collapse**.
- arc adds signal → **"The Surprisal Arc"** paper.
- arc still null → **"AI overshoots UID: level & uniformity, not arc shape, are the tell"** paper
  (already supported by this pilot).
Both outcomes are publishable — no losing hypothesis.

## Status of infrastructure (works)
Surprisal extraction (gpt2/MPS, ~27 texts/s), arc-feature pipeline, HC3 loader, 5-fold CV, and
figure output all run end-to-end. Only the data **regime/length** needs to change.
