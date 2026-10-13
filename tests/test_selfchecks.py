"""Self-checks that the paper relies on (run: python -m pytest tests -q)."""
import numpy as np
from tdiag import decomp, lfr, nonuniform, stats
from tdiag.metrics import ssim

rng = np.random.default_rng(0)


def _pair(h=96, w=96):
    y = np.clip(rng.random((h, w)) * .6 + .2, 0, 1)
    x = np.clip(.35 * y + .3 + rng.normal(0, .01, y.shape), 0, 1)
    return x, y


def test_oracle_affine_share_is_one_and_struct_zero():
    x, y = _pair(); o = decomp.apply_affine(x, y)
    r = decomp.decompose(x, y, o)
    assert abs(r["gain_struct"]) < 1e-6 and abs(r["gain_total"] - r["gain_affine"]) < 1e-6


def test_struct_gain_invariant_to_global_affine_of_output():
    x, y = _pair(); o = np.clip(y + rng.normal(0, .02, y.shape), 0, 1)
    g1 = decomp.decompose(x, y, o)["gain_struct"]
    g2 = decomp.decompose(x, y, np.clip(.8 * o + .05, 0, 1))["gain_struct"]
    assert abs(g1 - g2) < 0.05          # equal up to clipping


def test_pyramid_is_invertible():
    x, _ = _pair(); assert np.allclose(sum(lfr.pyramid(x)), x, atol=1e-5)


def test_lfr_matches_bruteforce_regression():
    x, y = _pair(128, 128)
    for k, (bx, by) in enumerate(zip(lfr.pyramid(x), lfr.pyramid(y))):
        A = np.stack([bx.ravel(), np.ones(bx.size)], 1)
        coef = np.linalg.lstsq(A, by.ravel(), rcond=None)[0]
        mse_best = np.mean((A @ coef - by.ravel()) ** 2)
        ref = 10 * np.log10(np.mean((bx - by) ** 2) / mse_best)
        assert abs(lfr.band_ceiling(x, y)[k]["ceiling_db"] - ref) < 1e-3


def test_nonuniformity_zero_for_uniform_affine_degradation():
    y = rng.random((64, 64)); x = .4 * y + .3
    assert nonuniformity_ok(nonuniform.nonuniformity(x, y)["NU"])


def nonuniformity_ok(v):
    return abs(v) < 1e-3


def test_ssim_identity():
    _, y = _pair(); assert abs(ssim(y, y) - 1) < 1e-9


def test_ratio_bootstrap_brackets_point_estimate():
    num = rng.normal(1, .1, 60); den = rng.normal(2, .1, 60); cl = np.repeat(np.arange(20), 3)
    est, lo, hi = stats.cluster_bootstrap_ratio(num, den, cl, n_boot=500)
    assert lo <= est <= hi


def test_closed_forms_and_identity():
    """G_aff and G_struct match their correlation closed forms; G_total = G_aff + G_struct - Delta_o."""
    r = np.random.default_rng(3)
    y = 0.2 + 0.6 * r.random((96, 96))
    x = 0.5 * y + 0.25 + 0.02 * r.standard_normal(y.shape)          # stays inside [0, 1]: no clipping
    o = 0.8 * y + 0.05 + 0.02 * r.standard_normal(y.shape)
    d = decomp.decompose(x, y, o)
    rho = lambda a: np.corrcoef(a.ravel(), y.ravel())[0, 1]
    g_aff = 10 * np.log10(((x - y) ** 2).mean() / (y.var() * (1 - rho(x) ** 2)))
    g_struct = 10 * np.log10((1 - rho(x) ** 2) / (1 - rho(o) ** 2))
    assert abs(d['gain_affine'] - g_aff) < 1e-6 and abs(d['gain_struct'] - g_struct) < 1e-6
    delta_o = d['psnr_affine_out'] - d['psnr_out']
    assert delta_o >= -1e-9 and abs(d['gain_total'] - (d['gain_affine'] + d['gain_struct'] - delta_o)) < 1e-9
