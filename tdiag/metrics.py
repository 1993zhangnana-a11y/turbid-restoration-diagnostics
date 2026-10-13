"""SSIM (Wang et al., 2004; bit-identical to skimage with gaussian_weights, sigma=1.5) and optional
LPIPS / BRISQUE wrappers. LPIPS needs `pip install lpips`; BRISQUE needs `pip install piq`."""
from __future__ import annotations
import numpy as np


def _gauss_filter(img, k):
    try:
        from scipy.ndimage import correlate1d
        a = correlate1d(img, k, axis=0, mode="reflect")
        return correlate1d(a, k, axis=1, mode="reflect")
    except Exception:  # pragma: no cover
        p = len(k) // 2
        a = np.pad(img, ((p, p), (p, p)), mode="reflect")
        tmp = np.stack([(a[:, j:j + 2 * p + 1] * k).sum(1) for j in range(img.shape[1])], 1)
        return np.stack([(tmp[i:i + 2 * p + 1, :] * k[:, None]).sum(0) for i in range(img.shape[0])], 0)


def ssim(x, y, mask=None, sigma=1.5, L=1.0):
    r = int(3.5 * sigma + .5); n = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-n ** 2 / (2 * sigma ** 2)); k /= k.sum()
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    ux, uy = _gauss_filter(x, k), _gauss_filter(y, k)
    uxx, uyy, uxy = _gauss_filter(x * x, k), _gauss_filter(y * y, k), _gauss_filter(x * y, k)
    vx, vy, vxy = uxx - ux * ux, uyy - uy * uy, uxy - ux * uy
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
    S = S[r:-r, r:-r]
    if mask is not None:
        mm = np.asarray(mask, bool)[r:-r, r:-r]
        return float(S[mm].mean()) if mm.sum() >= 16 else float(S.mean())
    return float(S.mean())


class LPIPS:
    def __init__(self, net="alex", device="cpu"):
        import torch, lpips  # noqa: F401
        self.torch = torch; self.dev = device
        self.net = lpips.LPIPS(net=net, verbose=False).to(device)

    def __call__(self, p, y):
        t = lambda z: self.torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1))[None] * 2 - 1).float().to(self.dev)
        with self.torch.inference_mode():
            return float(self.net(t(p), t(y)).item())


def brisque(p):
    import torch, piq
    t = torch.from_numpy(np.ascontiguousarray(p.transpose(2, 0, 1))[None]).float().clamp(0, 1)
    with torch.inference_mode():
        return float(piq.brisque(t, data_range=1.).item())
