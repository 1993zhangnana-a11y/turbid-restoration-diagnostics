"""Paired statistics used throughout the paper.

* cluster_bootstrap_mean: resample whole clusters (e.g. scene = pose x background) with replacement.
  Images of the same scene are strongly correlated; resampling images independently gives spuriously
  narrow intervals.
* cluster_bootstrap_ratio: same, for the ratio-of-means statistic `share`.
* seed_sign_consistency: do all per-seed means share one sign?
"""
from __future__ import annotations
import numpy as np


def _groups(values, clusters):
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    keys = np.unique(clusters)
    return [values[clusters == k] for k in keys]


def cluster_bootstrap_mean(values, clusters, n_boot=4000, seed=0, level=95):
    g = _groups(values, clusters); k = len(g); rng = np.random.default_rng(seed)
    bs = np.array([np.nanmean(np.concatenate([g[i] for i in rng.integers(0, k, k)])) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(bs, [(100 - level) / 2, 100 - (100 - level) / 2])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(np.nanmean(values)), float(lo), float(hi), float(max(p, 1.0 / n_boot))


def cluster_bootstrap_ratio(num, den, clusters, n_boot=4000, seed=0, level=95):
    num = np.asarray(num, float); den = np.asarray(den, float); clusters = np.asarray(clusters)
    keys = np.unique(clusters); idx = [np.where(clusters == k)[0] for k in keys]
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        out.append(num[sel].mean() / den[sel].mean())
    lo, hi = np.nanpercentile(out, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(num.mean() / den.mean()), float(lo), float(hi)


def seed_sign_consistency(values, seeds) -> bool:
    values = np.asarray(values, float); seeds = np.asarray(seeds)
    means = [np.nanmean(values[seeds == s]) for s in np.unique(seeds)]
    return len(set(np.sign(means))) == 1
