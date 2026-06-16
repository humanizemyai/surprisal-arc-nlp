# The Surprisal Arc — computational-linguistics study

**Finding (negative + mechanism): there is no aesthetic surprisal arc.** The *shape* of a text's
token-surprisal trajectory predicts neither human-vs-AI authorship nor human-judged writing quality.
What carries signal is the *level* and *variance* of surprisal — consistent with Uniform Information
Density, not "managed-surprise" aesthetics.

Research by **Berk Ustun**. Independent public benchmarks only (no in-house data). Full write-up in
[`paper/paper.md`](paper/paper.md); results ledger in [`tables/SUMMARY.md`](tables/SUMMARY.md).

## TL;DR results (5-fold ROC-AUC; LM = gpt2, replicated with Pythia-410m)
- **E1 (HC3, detection):** mean-perplexity 0.97, burstiness 0.95, **arc-shape 0.55 (chance)**.
- **E2 (MAGE, long-form + paraphrase attack):** **arc-shape 0.51–0.53** (chance) in both conditions;
  paraphrase **collapses mean-perplexity 0.80→0.58** but **leaves burstiness 0.80→0.78** → burstiness is
  the attack-robust statistic. Holds per-domain (cnn/dialogsum/imdb/pubmed).
- **E3 (ASAP-AES, quality):** arc adds nothing (ΔCV-R² −0.001, nested-F p=0.08); length dominates;
  more-uniform / lower-perplexity essays score *higher* (UID direction, opposite of the aesthetic arc).

## What we tested
Each text → per-token surprisal `s_i = -log2 p(x_i | x_<i)`. We isolate the **arc shape** as the
low-frequency FFT of the length-normalized, smoothed, z-normalized surprisal curve (mean & variance
removed), and pit it against **mean** (perplexity) and **burstiness** (variance) baselines. The
aesthetic view predicts the shape should matter; UID predicts uniformity should. UID wins.

## Data (external, independent, multi-author)
- **HC3** (Guo et al. 2023) — human vs ChatGPT (`open_qa`).
- **MAGE** (Li et al. 2024) — GPT-4 OOD, raw + **paraphrased**, long-form, 4 domains.
- **ASAP-AES** (Hewlett) — essays with independent human holistic scores (long sets 1/2/8).

## Reproduce
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
# canonical driver — all 3 experiments for one surprisal LM:
ARC_LM=gpt2                  .venv/bin/python src/run_all.py   # -> tables/results_gpt2.json
ARC_LM=EleutherAI/pythia-410m .venv/bin/python src/run_all.py # robustness -> tables/results_*.json
# individual experiments (also write per-experiment figures):
.venv/bin/python src/pilot/run_pilot.py        # E1 HC3      -> figures/pilot_arc_vs_burstiness.png
.venv/bin/python src/detect/run_decider.py     # E2 MAGE     -> figures/decider_arc_paraphrase.png
.venv/bin/python src/stats/run_quality.py      # E3 ASAP-AES -> figures/quality_arc.png
```

## Layout
`config/` params · `src/{data,surprisal,arc,detect,stats,pilot,run_all.py}` pipeline ·
`tables/` results + findings notes · `figures/` · `paper/` preprint
