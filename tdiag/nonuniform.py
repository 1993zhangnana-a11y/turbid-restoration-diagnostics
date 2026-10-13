"""Global, per-channel and tiled oracle affine; spatial non-uniformity index.

NU = 1 - G_aff_global / G_aff_tile, where G_aff_tile is the oracle gain when every tile of a T x T grid gets
its own two-parameter affine. NU = 0: one global affine is as good as a spatially varying one (uniform
degradation); NU -> 1: almost all affine-recoverable error needs a spatially varying correction.
"""
from __future__ import annotations
import numpy as np
from .decomp import affine_fit, psnr


def global_affine(x, y):
    a, c = affine_fit(x, y); return np.clip(a * x + c, 0, 1)


def channel_affine(x, y):
    o = np.empty_like(np.asarray(x, np.float64))
    for ch in range(x.shape[-1]):
        a, c = affine_fit(x[..., ch], y[..., ch]); o[..., ch] = np.clip(a * x[..., ch] + c, 0, 1)
    return o


def tiled_affine(x, y, T=4):
    H, W = x.shape[:2]; o = np.empty_like(np.asarray(x, np.float64))
    hs = np.linspace(0, H, T + 1).astype(int); ws = np.linspace(0, W, T + 1).astype(int)
    for i in range(T):
        for j in range(T):
            sl = (slice(hs[i], hs[i + 1]), slice(ws[j], ws[j + 1]))
            a, c = affine_fit(x[sl], y[sl]); o[sl] = np.clip(a * x[sl] + c, 0, 1)
    return o


def nonuniformity(x, y, T=4) -> dict:
    p_in = psnr(x, y)
    g_glob = psnr(global_affine(x, y), y) - p_in
    g_tile = psnr(tiled_affine(x, y, T), y) - p_in
    return dict(g_aff_global=g_glob, g_aff_tile=g_tile,
                NU=float(1 - g_glob / g_tile) if g_tile > 1e-6 else float("nan"))
