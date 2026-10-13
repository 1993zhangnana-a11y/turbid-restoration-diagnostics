# turbid-restoration-diagnostics

Code for the paper

> **Photometry or Structure? A Diagnostic Evaluation of Image Restoration in Turbid Media**
> Na-Na Zhang, Yaowen Guo, Junlin Li. *Submitted to IEEE Transactions on Image Processing*, 2026.

The paper does not propose a new restoration network. It provides method-agnostic measurement operators that
tell you **what a reported PSNR gain is made of**, **at which spatial scales it arises**, and **whether an
injected physical prior really helps**. This repository contains the measurement toolkit (`tdiag`), the
networks and prior-injection modules used in the study (`models`), the experiment drivers (`experiments`)
and the plotting scripts for every figure (`figures`).

## What the toolkit measures

Given degraded `x`, clear `y` and restored `o` images (any method, any dataset):

| Quantity | Definition | Meaning |
|---|---|---|
| `G_total`  | PSNR(o,y) − PSNR(x,y) | the usually reported gain |
| `G_aff`    | PSNR(A(x),y) − PSNR(x,y), A = least-squares global affine to `y` | what a two-parameter photometric alignment already achieves (oracle) |
| `G_struct` | PSNR(A(o),y) − PSNR(A(x),y) | affine-invariant structural gain |
| `share`    | mean(G_aff) / mean(G_total) | fraction of the gain that is photometric (ratio of means) |
| `LFR_k`    | 10 log10( E[(x_k−y_k)²] / (var(y_k)(1−ρ_k²)) ) | bound of filters with one scalar gain + offset per band `k` |
| headroom   | LFR_k − measured band gain | negative = the method uses cross-band / non-linear information |
| `NU`       | 1 − G_aff,global / G_aff,tiled | spatial non-uniformity of the degradation |

All confidence intervals are **cluster bootstraps** (clusters = scenes) and shares are bootstrapped as ratios
of means.

## Installation

```bash
git clone https://github.com/1993zhangnana-a11y/turbid-restoration-diagnostics.git
cd turbid-restoration-diagnostics
pip install -r requirements.txt        # numpy/scipy/pillow/pandas/matplotlib; torch/lpips/piq are optional
```

The measurement toolkit needs only numpy, scipy and Pillow. PyTorch is needed only to train networks.

## Quick start: decompose the gains of your own method

```bash
# directory mode: files with the same relative name are paired
python -m tdiag.cli --degraded data/hazy --clear data/gt --restored results/mymethod \
                    --out runs/mymethod --size 0 --rgb            # native resolution, RGB (literature protocol)

# manifest mode: CSV with columns degraded,clear,restored[,group][,cluster]
python -m tdiag.cli --manifest pairs.csv --out runs/mymethod
```

Outputs: `per_image.csv` (all quantities per image), `summary.csv` (means, 95 % cluster-bootstrap CIs,
`share`), `bands.csv` (per-band ρ, LFR, measured gain and headroom).

From Python:

```python
from tdiag import decompose, band_ceiling, band_gain, nonuniformity
r = decompose(x, y, o, mask)          # dict: psnr_in, gain_affine, gain_total, gain_struct, ...
lfr = band_ceiling(x_luma, y_luma)    # per-band rho and LFR (dB)
nu = nonuniformity(x, y, T=4)         # global vs tiled oracle affine, NU index
```

## Reproducing the public-benchmark results (Section V-B/C and VI)

Download the NTIRE archives of Dense-Haze, NH-HAZE, O-HAZE and I-HAZE from the challenge pages
(data.vision.ee.ethz.ch, NTIRE 2018/2019/2020) and unpack each into `<root>/<NAME>/_raw/`
(`<NAME>` ∈ `DENSE-HAZE, NH-HAZE, O-HAZE, I-HAZE`). Then

```bash
python experiments/public_benchmarks.py --root <root> --out runs/public --tiles 4,8 --folds 5 --iters 6000
python figures/fig_public.py runs/public runs/figs/fig_public
```

Part A (no training) measures, on the official test splits at native resolution in RGB, the oracle global,
per-channel and tiled affine gains and the non-uniformity index. Part B trains the paper's NAFNet-Lite by
5-fold cross-validation on each benchmark (about 70 min per benchmark on one RTX 5080) so that every image has
an out-of-fold restoration to decompose.

## Prior-effect budget curves (Section VII)

`experiments/budget_curve.py` trains each prior arm once and evaluates it in situ at 14 budgets, for any of the
three backbone families (`nafnet`, `restormer`, `resunet`). It reads a dataset manifest (CSV with
`split, degraded, clear, level, scene, geometry, tau, b`; see the script header). Significance:
`experiments/signif.py gain_per_unit.csv signif.csv`, and `experiments/summarise_end.py signif.csv W:none background 80` gives the four-level end-of-training effect (Table V). Wiener coefficients are estimated with
`models/wiener.py:estimate` on training data only.

The in-house turbid-medium data used in the paper are not public at this time; the drivers
are provided so that the protocol can be applied to other data.

## Models

| Family | Configuration in the paper | Builder |
|---|---|---|
| gated convolution | NAFNet-Lite (2,2,4), 0.905 M with prior heads | `build_core('nafnet', 3, 32, (2,2,4))` |
| channel attention | Restormer-tiny (1,3,4), 1.257 M | `build_core('restormer', 3, 32, (1,3,4))` |
| plain residual conv. | ResUNet (2,2,1), 0.938 M | `build_core('resunet', 3, 32, (2,2,1))` |

`models.build_final_model()` is the finalised baseline (828,549 parameters). `models.PriorWrapper` injects
tool geometry at the bottleneck; with zero geometry it reduces to the plain core.

## Figures

| Paper figure | Script | Input |
|---|---|---|
| Overview | `figures/fig1_overview.py` | none (schematic) |
| Public benchmarks, NU | `figures/fig_public.py` | output of `experiments/public_benchmarks.py` |
| LFR headroom | `figures/fig_lfr.py` | `table_lfr.csv` of the held-out evaluation |
| Budget × architecture | `figures/fig_backbones.py` | `signif.csv` of each family |
| Controls (capacity, parameters, annealing) | `figures/fig_controls.py` | list of `signif.csv` |
| Metric dependence | `figures/fig_metrics.py` | `table_metrics.csv` of the held-out evaluation |
| Cross-session comparison | `figures/fig_sessions.py` | `table_sessions.csv` |

`figures/figstyle.py` sets fonts (Times by default), embeds TrueType fonts and exports PNG/PDF/EPS
(EPS via `pdftops` so that transparent confidence bands survive).

## Tests

```bash
python -m pytest tests -q        # or: python tests/test_selfchecks.py ; python tests/test_models.py
```

The self-checks verify the properties stated in the paper: the oracle affine has share 1 and zero
structural gain, `G_struct` is invariant to global affine maps of the output, the pyramid is exactly
invertible, the closed-form LFR equals a brute-force per-band regression, and the parameter counts of
all seven network configurations.

## Citation

```bibtex
@article{zhang2026photometry,
  title   = {Photometry or Structure? A Diagnostic Evaluation of Image Restoration in Turbid Media},
  author  = {Zhang, Na-Na and Guo, Yaowen and Li, Junlin},
  journal = {IEEE Transactions on Image Processing},
  note    = {submitted},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
