"""Turn a per-token surprisal sequence into baseline + arc-shape features.

Baselines (mean = perplexity proxy, std/mad = burstiness proxy) are the magnitude/variance
information the arc-shape deliberately discards. Arc-shape = low-frequency FFT magnitudes of
the z-normalized, resampled, smoothed surprisal curve -> pure SHAPE, independent of mean+variance.
"""
from __future__ import annotations

import numpy as np


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size <= w:
        return x
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, kernel, mode="same")


def arc_curve(surprisals: np.ndarray, resample_len: int = 64,
              smooth_window: int = 5) -> np.ndarray:
    """Resample surprisal to fixed length, then low-pass smooth -> T(t) (not normalized)."""
    s = np.asarray(surprisals, dtype=np.float64)
    if s.size == 0:
        return np.zeros(resample_len)
    if s.size == 1:
        return np.full(resample_len, float(s[0]))
    xp = np.linspace(0.0, 1.0, num=s.size)
    xq = np.linspace(0.0, 1.0, num=resample_len)
    return _moving_average(np.interp(xq, xp, s), smooth_window)


def normalized_arc(surprisals: np.ndarray, resample_len: int = 64,
                   smooth_window: int = 5) -> np.ndarray:
    """Arc curve z-normalized to pure shape (mean 0, unit std)."""
    c = arc_curve(surprisals, resample_len, smooth_window)
    c = c - c.mean()
    sd = c.std()
    return c / sd if sd > 1e-8 else c


def baseline_features(surprisals: np.ndarray) -> dict:
    """Magnitude/variance features (what arc-shape throws away)."""
    s = np.asarray(surprisals, dtype=np.float64)
    if s.size == 0:
        return {"mean": 0.0, "std": 0.0, "mad": 0.0}
    mad = float(np.mean(np.abs(np.diff(s)))) if s.size > 1 else 0.0  # local variation
    return {"mean": float(s.mean()), "std": float(s.std()), "mad": mad}


def arc_shape_features(surprisals: np.ndarray, resample_len: int = 64,
                       smooth_window: int = 5, n_fft_coeffs: int = 8) -> np.ndarray:
    """Pure SHAPE: low-frequency FFT magnitudes of the z-normalized arc (DC dropped)."""
    c = normalized_arc(surprisals, resample_len, smooth_window)
    mags = np.abs(np.fft.rfft(c))            # length resample_len//2 + 1
    feats = mags[1:1 + n_fft_coeffs]         # skip DC (index 0 ~ 0 after centering)
    if feats.size < n_fft_coeffs:
        feats = np.pad(feats, (0, n_fft_coeffs - feats.size))
    return feats.astype(np.float64)


def band_energy_features(surprisals: np.ndarray, resample_len: int = 128,
                         n_bands: int = 8) -> np.ndarray:
    """Relative power in n_bands equal frequency bands of the surprisal spectrum (DC dropped).

    The full 'EQ' distribution across scales — global arc (low band) ... token jitter (high band) —
    rather than only the single low band the arc-shape feature used. Sums to 1, so it captures the
    DISTRIBUTION across bands independent of total variance (burstiness). This is the DJ-EQ / E(tau)
    band-energy idea ported to the surprisal time-series.
    """
    s = np.asarray(surprisals, dtype=np.float64)
    if s.size < n_bands + 2:
        return np.zeros(n_bands)
    xp = np.linspace(0.0, 1.0, num=s.size)
    xq = np.linspace(0.0, 1.0, num=resample_len)
    r = np.interp(xq, xp, s)
    r = r - r.mean()
    power = np.abs(np.fft.rfft(r)) ** 2
    power = power[1:]                         # drop DC
    if power.sum() <= 0:
        return np.zeros(n_bands)
    edges = np.linspace(0, power.size, n_bands + 1).astype(int)
    bands = np.array([power[edges[i]:edges[i + 1]].sum() for i in range(n_bands)], dtype=np.float64)
    tot = bands.sum()
    return bands / tot if tot > 0 else bands
