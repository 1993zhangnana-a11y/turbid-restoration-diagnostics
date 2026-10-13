"""Affine--structure decomposition of a restoration gain.

Given a degraded input x, a clear reference y, a restored output o and a pixel mask m:

  G_total  = PSNR(o, y) - PSNR(x, y)
  G_aff    = PSNR(A(x), y) - PSNR(x, y)          A(z) = a_z * z + c_z, least squares to y on m
  G_struct = PSNR(A(o), y) - PSNR(A(x), y)
  share    = mean(G_aff) / mean(G_total)          (ratio of means, NOT mean of ratios)

G_aff is an *oracle* (uses y) and is the "trivially attainable" reference. G_struct is invariant to any
global affine transform of o, so G_struct == 0 for any global photometric correction.

Closed forms (no clipping; rho_z = Pearson correlation of z and y on m, peak 1):
  G_aff    = 10 log10( MSE(x, y) / (var(y) (1 - rho_x^2)) )
  G_struct = 10 log10( (1 - rho_x^2) / (1 - rho_o^2) )
  G_total  = G_aff + G_struct - Delta_o,  Delta_o = PSNR(A(o), y) - PSNR(o, y) >= 0
"""
from __future__ import annotations
import numpy as np


def psnr(a, b, m=None, peak: float = 1.0) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = (a - b) ** 2
    if m is not None:
        d = d[np.asarray(m, bool)]
    e = float(d.mean()) if d.size else float("nan")
    return float(10 * np.log10(peak * peak / max(e, 1e-12)))


def affine_fit(x, y, m=None):
    """argmin_{a,c} || a*x + c - y ||^2 over mask m (all channels pooled)."""
    xv = np.asarray(x, np.float64); yv = np.asarray(y, np.float64)
    if m is not None:
        mm = np.asarray(m, bool)
        if xv.ndim == 3 and mm.ndim == 2:
            mm = np.repeat(mm[..., None], xv.shape[-1], -1)
        xv, yv = xv[mm], yv[mm]
    xv = xv.ravel(); yv = yv.ravel()
    v = xv.var()
    if v < 1e-12:
        return 1.0, float(yv.mean() - xv.mean())
    a = float(((xv - xv.mean()) * (yv - yv.mean())).mean() / v)
    return a, float(yv.mean() - a * xv.mean())


def apply_affine(z, y, m=None, clip=True):
    a, c = affine_fit(z, y, m)
    o = a * np.asarray(z, np.float64) + c
    return np.clip(o, 0, 1) if clip else o


def decompose(x, y, o=None, m=None) -> dict:
    """Per-image decomposition. Images in [0,1], any matching shape (H,W) or (H,W,C)."""
    p_in = psnr(x, y, m)
    p_aff_in = psnr(apply_affine(x, y, m), y, m)
    r = dict(psnr_in=p_in, psnr_affine_in=p_aff_in, gain_affine=p_aff_in - p_in)
    if o is not None:
        p_out = psnr(o, y, m)
        p_aff_out = psnr(apply_affine(o, y, m), y, m)
        r.update(psnr_out=p_out, psnr_affine_out=p_aff_out, gain_total=p_out - p_in,
                 gain_struct=p_aff_out - p_aff_in)
    return r


def share_of_means(g_aff, g_total) -> float:
    g_aff = np.asarray(g_aff, float); g_total = np.asarray(g_total, float)
    den = np.nanmean(g_total)
    return float(np.nanmean(g_aff) / den) if abs(den) > 1e-12 else float("nan")
