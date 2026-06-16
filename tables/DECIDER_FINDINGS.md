# Decider findings — MAGE GPT4 OOD (long-form), raw vs paraphrased

## Result (5-fold ROC-AUC; LM = gpt2; domains cnn/dialogsum/imdb/pubmed; 250/cell; median ~205 words)
| condition                         | mean-ppl | burstiness | arc-shape | mean+burst | all  |
|-----------------------------------|----------|------------|-----------|------------|------|
| A_raw  (human vs GPT4)            | 0.803    | 0.805      | 0.511     | 0.852      | 0.843|
| B_para (human vs GPT4-paraphrased)| **0.595**| **0.789**  | 0.529     | 0.856      | 0.848|

Figure: `figures/decider_arc_paraphrase.png` — the three mean arcs (human / GPT4 / GPT4-para) overlap.

## Verdict on the arc hypothesis: FALSIFIED
Arc-shape is at chance in **both** regimes (HC3 easy + MAGE hard) and **both** length scales
(~100w + ~205w), as a detector **and** descriptively (the normalized arcs overlap). Having controlled
the two confounds we suspected (length, and a regime where mean-perplexity demonstrably collapses),
the arc still carries no signal. The surprisal *arc shape* is not a human/AI tell.

## The real finding (counter-intuitive, publishable)
Paraphrase is a **selective** attack on surprisal statistics:
- it **collapses mean-perplexity** (0.803 -> 0.595) — it humanizes the *level* of predictability;
- it **does NOT collapse burstiness** (0.805 -> 0.789) — surprisal *variance* survives the attack;
- so **mean+burst stays flat at ~0.85** under attack (burstiness compensates when level fails).

=> Burstiness (surprisal variance) is the attack-robust statistic vs generic paraphrase.

## Connection to the in-house humanizer (precise, no conflation)
In-house, burstiness AUC ~0.42 for predicting **GPTZero-pass** on the vocabulary-tuned humanizer —
a *different label* (detector-pass) and a *different attack* (vocab elevation) than here (human-vs-AI,
generic paraphrase, AUC 0.79). Together they suggest the effective lever is **attack-specific**:
generic paraphrase defeats *level*, vocab-tuning defeats the *vocabulary/GPTZero* axis. A map of
"which humanization attack defeats which statistic" is the real contribution.

## Caveats (fix in the full study)
- Domain composition pooled (4 domains, capped 250/cell), not stratified -> add per-domain control.
- Single LM (gpt2) for surprisal -> add a second (Pythia/Llama) for robustness.
- One attack (MAGE paraphrase). RAID's 11 attacks would give a full robustness map.

## Reframed direction (was "Surprisal Arc")
"Which surprisal statistics survive humanization attacks?" — mean-perplexity is fragile, burstiness
is robust; map attack x statistic on public benchmarks (MAGE + RAID's 11 attacks). Public paper +
private humanizer application. The arc is reported as an honest, well-controlled negative result.
