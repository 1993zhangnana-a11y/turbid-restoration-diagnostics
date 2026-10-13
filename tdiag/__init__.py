"""tdiag -- diagnostic evaluation of image restoration in turbid media.

Core measurement operators (numpy only):
  decomp     affine--structure decomposition of a PSNR gain (G_total, G_aff, G_struct, share)
  lfr        band-wise linear filter reference (LFR) on a box-Laplacian pyramid
  nonuniform tiled-affine gain and the spatial non-uniformity index NU
  stats      paired, cluster-bootstrap confidence intervals and seed sign consistency
  metrics    SSIM (Wang et al. 2004, matches skimage) and optional LPIPS / BRISQUE wrappers
"""
from .decomp import psnr, affine_fit, apply_affine, decompose, share_of_means
from .lfr import BANDS, BAND_NAMES, pyramid, band_ceiling, band_gain
from .nonuniform import tiled_affine, channel_affine, global_affine, nonuniformity
from .stats import cluster_bootstrap_mean, cluster_bootstrap_ratio, seed_sign_consistency

__version__ = "1.0.0"
