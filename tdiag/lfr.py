"""Band-wise linear filter reference (LFR).

Box-blur Laplacian pyramid: L_0 = z, L_k = box_{s_k}(L_{k-1}), band_k = L_{k-1} - L_k (k=1..5), DC = L_5.
The decomposition is exactly invertible (bands sum to z).

For estimators of the form  y_hat_k = a_k x_k + c_k  (one scalar gain plus offset per band) the minimum
MSE is var(y_k) (1 - rho_k^2), so the attainable gain over doing nothing is

    LFR_k = 10 log10( E[(x_k - y_k)^2] / (var(y_k) (1 - rho_k^2)) ).

LFR bounds this *linear filter family* only (it is the bound of an ideally calibrated band-wise scalar
Wiener filter). Non-linear or cross-band methods may exceed it; the excess is a diagnostic. Bands are not
orthogonal, so LFR is a per-band statement and must not be summed over bands.
"""
from __future__ import annotations
import numpy as np

BANDS = (2, 4, 8, 16, 32)
BAND_NAMES = ["<2px", "2-4px", "4-8px", "8-16px", "16-32px", ">32px(DC)"]


def box(a, k):
    """k x k mean filter with reflect boundary (scipy if available, else integral image)."""
    try:
        import scipy.ndimage as nd
        return nd.uniform_filter(np.asarray(a, np.float32), size=k, mode="reflect")
    except Exception:  # pragma: no cover
        pad = k // 2
        b = np.pad(a, pad, mode="reflect")
        c = np.pad(np.cumsum(np.cumsum(b, 0), 1), ((1, 0), (1, 0)))
        H, W = a.shape
        return ((c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W]) / (k * k)).astype(np.float32)


def pyramid(a, sizes=BANDS):
    out, prev = [], np.asarray(a, np.float32)
    for s in sizes:
        lo = box(prev, s)
        out.append(prev - lo)
        prev = lo
    out.append(prev)
    return out


def band_ceiling(x, y, m=None, sizes=BANDS):
    """Per band: rho_k, current MSE, LFR floor and LFR_k (dB). x, y are 2-D (luma)."""
    px, py = pyramid(x, sizes), pyramid(y, sizes)
    m = np.ones(np.shape(x), bool) if m is None else np.asarray(m, bool)
    rows = []
    for k in range(len(px)):
        xv, yv = px[k][m].astype(np.float64), py[k][m].astype(np.float64)
        sx, sy = xv.std(), yv.std()
        rho = float(np.corrcoef(xv, yv)[0, 1]) if sx > 1e-9 and sy > 1e-9 else 0.0
        rho = 0.0 if not np.isfinite(rho) else rho
        cur = float(np.mean((xv - yv) ** 2))
        floor = float(yv.var() * max(1 - rho * rho, 1e-12))
        rows.append(dict(band=BAND_NAMES[k], rho=rho, mse_now=cur, mse_floor=floor,
                         ceiling_db=float(10 * np.log10(max(cur, 1e-12) / max(floor, 1e-12)))))
    return rows


def band_gain(x, o, y, m=None, sizes=BANDS):
    """Measured per-band gain (dB) of output o over input x."""
    px, po, py = pyramid(x, sizes), pyramid(o, sizes), pyramid(y, sizes)
    m = np.ones(np.shape(x), bool) if m is None else np.asarray(m, bool)
    g = []
    for k in range(len(px)):
        ein = float(np.mean((px[k][m] - py[k][m]) ** 2)); eo = float(np.mean((po[k][m] - py[k][m]) ** 2))
        g.append(float(10 * np.log10(max(ein, 1e-12) / max(eo, 1e-12))))
    return g
