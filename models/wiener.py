"""Band-wise scalar Wiener pre-filter (the 'pre-filter' prior of the paper).

Bands: difference-of-box decomposition of the input at sizes (2,4,8,16,32) px plus a DC residual
(bands sum exactly to the image). Coefficients are estimated per degradation level on TRAINING data only:
  S_k  clear-image band power
  E1_k single-frame error power, EK_k  K-frame-averaged error power
  N_k  = (E1_k - EK_k) K / (K-1)            per-frame random-noise power in band k
  G_k  = 1 - sqrt(max(E1_k - N_k, 0) / S_k)  effective band transfer of the degradation
  w_k  = G_k S_k / (G_k^2 S_k + N_k), clipped to [0.1, 12]
The DC band uses an affine fit (a_DC, c_DC), because veiling light is an offset.
"""
import numpy as np

SIZES = (2, 4, 8, 16, 32)


def bands(a, sizes=SIZES):
    import scipy.ndimage as nd
    out, prev = [], a
    for sz in sizes:
        lo = nd.uniform_filter(a, size=sz); out.append(prev - lo); prev = lo
    out.append(prev)
    return out


def estimate(clear_y, single_y, avg_y, masks, K):
    """clear_y/single_y/avg_y: lists of 2-D luma images in [0,1]; masks: list of bool masks; K: frames averaged."""
    NB = len(SIZES) + 1; A = np.zeros((7, NB))
    for c, x1, xk, m in zip(clear_y, single_y, avg_y, masks):
        cb, ib = bands(c), bands(x1); e1, ek = bands(x1 - c), bands(xk - c)
        for k in range(NB):
            A[0, k] += np.mean(cb[k][m] ** 2); A[1, k] += np.mean(e1[k][m] ** 2); A[2, k] += np.mean(ek[k][m] ** 2)
            A[3, k] += np.mean(ib[k][m]); A[4, k] += np.mean(cb[k][m]); A[5, k] += np.mean(ib[k][m] ** 2)
            A[6, k] += np.mean(ib[k][m] * cb[k][m])
    A /= max(len(clear_y), 1)
    S, E1, EK, mx, mc, mxx, mxc = A
    N = np.maximum((E1 - EK) * K / max(K - 1, 1), 1e-10) if K > 1 else np.full(NB, 1e-10)
    G = np.clip(1 - np.sqrt(np.maximum(E1 - N, 0) / np.maximum(S, 1e-12)), 1e-3, 2.0)
    w = np.clip(G * S / (G * G * S + N), 0.1, 12.0)
    var = max(float(mxx[-1] - mx[-1] ** 2), 1e-12)
    a_dc = float(np.clip((mxc[-1] - mx[-1] * mc[-1]) / var, 0.1, 12.0)); c_dc = float(mc[-1] - a_dc * mx[-1])
    return dict(w=w.tolist(), G=G.tolist(), N=N.tolist(), S=S.tolist(), a_dc=a_dc, c_dc=c_dc)


def apply_uint8(x, coeff):
    """x uint8 [H,W,C] -> filtered uint8 [H,W,C] with coefficients of one degradation level."""
    out = np.empty_like(x); w = coeff["w"]
    for c in range(x.shape[-1]):
        bb = bands(x[..., c].astype(np.float32) / 255)
        y = sum(w[k] * bb[k] for k in range(len(bb) - 1)) + coeff["a_dc"] * bb[-1] + coeff["c_dc"]
        out[..., c] = np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)
    return out
