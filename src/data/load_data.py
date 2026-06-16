"""Loaders for external, independent benchmarks.

Pilot uses HC3 (Guo et al. 2023) — human vs ChatGPT answers. No in-house data.
HC3 ships a (now-unsupported) loading script, so we pull the raw JSONL from the Hub directly.
"""
from __future__ import annotations

import json
import random


def load_hc3(config_name: str = "open_qa", n_per_class: int = 100,
             min_words: int = 40, seed: int = 42):
    """Return (texts, labels) with labels 0=human, 1=ChatGPT, balanced n_per_class each."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="Hello-SimpleAI/HC3",
                           filename=f"{config_name}.jsonl", repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)

    def ok(t) -> bool:
        return isinstance(t, str) and len(t.split()) >= min_words

    human: list[str] = []
    ai: list[str] = []
    for i in order:
        ex = rows[i]
        if len(human) < n_per_class:
            for a in (ex.get("human_answers") or []):
                if ok(a):
                    human.append(a.strip())
                    break
        if len(ai) < n_per_class:
            for a in (ex.get("chatgpt_answers") or []):
                if ok(a):
                    ai.append(a.strip())
                    break
        if len(human) >= n_per_class and len(ai) >= n_per_class:
            break

    texts = human + ai
    labels = [0] * len(human) + [1] * len(ai)
    return texts, labels


def load_mage_ood(min_words: int = 60, max_per_cell: int = 250, seed: int = 42):
    """MAGE GPT4 OOD pools for the paraphrase-attack decider (domain-matched, long-form).

    Returns {'human', 'ai_raw', 'ai_para', 'domains'}. Uses only domains present in all three
    conditions (human + raw GPT4 + paraphrased GPT4) so raw-vs-para is a clean controlled contrast.
    MAGE label: 0 = machine, 1 = human; src = '<domain>_<source>' (e.g. cnn_human / cnn_gpt4 / cnn_gpt4_para).
    """
    import random

    import pandas as pd
    from huggingface_hub import hf_hub_download

    pg = pd.read_csv(hf_hub_download("yaful/MAGE", "test_ood_set_gpt.csv", repo_type="dataset"))
    pp = pd.read_csv(hf_hub_download("yaful/MAGE", "test_ood_set_gpt_para.csv", repo_type="dataset"))

    def domain(s: str) -> str:
        return str(s).split("_")[0]

    human = pg[pg["src"].astype(str).str.endswith("_human")].copy()
    ai_raw = pg[pg["src"].astype(str).str.endswith("_gpt4")].copy()
    ai_para = pp[pp["src"].astype(str).str.endswith("_gpt4_para")].copy()
    for d in (human, ai_raw, ai_para):
        d["domain"] = d["src"].map(domain)

    doms = set(human["domain"]) & set(ai_raw["domain"]) & set(ai_para["domain"])

    def pick(df):
        sub = df[df["domain"].isin(doms)]
        rows = [(t, d) for t, d in zip(sub["text"].astype(str), sub["domain"])
                if len(t.split()) >= min_words]
        random.Random(seed).shuffle(rows)
        rows = rows[:max_per_cell]
        return [r[0] for r in rows], [r[1] for r in rows]

    h_t, h_d = pick(human)
    r_t, r_d = pick(ai_raw)
    p_t, p_d = pick(ai_para)
    return {"human": h_t, "ai_raw": r_t, "ai_para": p_t,
            "dom_human": h_d, "dom_ai_raw": r_d, "dom_ai_para": p_d,
            "domains": sorted(doms)}


def load_asap(sets=(1, 2, 8), n_per_set: int = 400, min_words: int = 120, seed: int = 42):
    """ASAP-AES essays with human holistic scores, long sets only (for the arc-quality arm).

    Returns a DataFrame [essay_set, text, score, wl]. domain1_score is the holistic human score;
    scales differ per set so callers z-score within set. Long sets 1/2/8 (median 365/368/626 words).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("TasfiaS/ASAP-AES", "training_set_rel3.tsv", repo_type="dataset")
    df = pd.read_csv(p, sep="\t", encoding="latin-1",
                     usecols=["essay_set", "essay", "domain1_score"])
    df = df.dropna(subset=["essay", "domain1_score"])
    df = df[df["essay_set"].isin(sets)].copy()
    df["text"] = df["essay"].astype(str)
    df["wl"] = df["text"].str.split().apply(len)
    df = df[df["wl"] >= min_words]

    parts = []
    for s in sets:
        sub = df[df["essay_set"] == s]
        if len(sub) > n_per_set:
            sub = sub.sample(n_per_set, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts).rename(columns={"domain1_score": "score"})
    return out[["essay_set", "text", "score", "wl"]].reset_index(drop=True)
