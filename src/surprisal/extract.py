"""Per-token surprisal s_i = -log2 p(x_i | x_<i) from an open causal LM.

Generalizes the perplexity probe in ~/humanizer-finetune/detector_ppl_test.py to a
full per-token surprisal sequence. Model is pluggable (gpt2 for the pilot).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_lm(model_name: str = "gpt2", device: str = "auto"):
    dev = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(dev)
    model.eval()
    return model, tok, dev


@torch.no_grad()
def token_surprisals(text: str, model, tok, device: str,
                     max_tokens: int = 512) -> np.ndarray:
    """Per-token surprisal in bits for tokens 2..n (token 1 has no left context)."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    ids = enc["input_ids"].to(device)
    if ids.shape[1] < 2:
        return np.array([], dtype=np.float32)
    logits = model(ids).logits  # (1, n, V)
    logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predict pos 1..n-1
    targets = ids[:, 1:]  # (1, n-1)
    tok_logp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
    return (-tok_logp / math.log(2.0)).cpu().numpy().astype(np.float32)


@torch.no_grad()
def windowed_surprisals(text: str, model, tok, device: str, window: int = 1024,
                        stride: int = 512, max_tokens_total: int = 16384) -> np.ndarray:
    """Per-token surprisal over a LONG text via sliding windows (for book-length passages).

    Non-first windows discard their first `stride` tokens (low left-context) to avoid
    double-counting and ensure each kept token has real context. Returns one ordered sequence.
    """
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens_total)["input_ids"][0]
    n = int(ids.shape[0])
    out = np.full(n, np.nan, dtype=np.float32)
    i = 0
    while i < n - 1:
        chunk = ids[i:i + window].unsqueeze(0).to(device)
        logits = model(chunk).logits
        lp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = chunk[:, 1:]
        sp = (-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0) / math.log(2.0)).cpu().numpy()
        keep_from = 0 if i == 0 else stride
        for k in range(keep_from, sp.shape[0]):
            pos = i + 1 + k
            if pos < n and np.isnan(out[pos]):
                out[pos] = sp[k]
        if i + window >= n:
            break
        i += stride
    return out[~np.isnan(out)].astype(np.float32)
