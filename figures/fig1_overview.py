# -*- coding: utf-8 -*-
"""Paper figure: overview schematic (no measured data).
Usage: python figures/fig1_overview.py OUT_PREFIX
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle

OUT = sys.argv[1] if len(sys.argv) > 1 else 'fig1_overview'

# low-saturation colors; distinguishable in grayscale
C_DATA = '#eef3f8'; E_DATA = '#5b7fa6'
C_NET  = '#f3eef8'; E_NET  = '#7a5ba6'
C_MEAS = '#eef8f1'; E_MEAS = '#3f8f63'
C_OUT  = '#fdf3e8'; E_OUT  = '#c1751f'
GREY   = '#555555'


def box(ax, x, y, w, h, title, body='', tag='', fc='#fff', ec='#666', lw=1.0,
        ts=8.4, bs=7.4, align='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    xc = x + w/2
    if body:
        ax.text(xc, y + h - 2.0, title, ha='center', va='top', fontsize=ts,
                fontweight='bold', zorder=3)
        ax.text(xc if align == 'center' else x + 1.6, y + h - 5.6, body,
                ha=align, va='top', fontsize=bs, color='#222', zorder=3, linespacing=1.45)
    else:
        ax.text(xc, y + h/2, title, ha='center', va='center', fontsize=ts,
                fontweight='bold', zorder=3)
    if tag:
        ax.text(x + w - 1.2, y + 1.0, tag, ha='right', va='bottom', fontsize=6.6,
                color=GREY, style='italic', zorder=3)


def arrow(ax, p, q, ls='-', color='#444', lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9,
                                 lw=lw, color=color, ls=ls, zorder=1,
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=1, shrinkB=1))


fig = plt.figure(figsize=(7.16, 3.95))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 55); ax.axis('off')

# ---------------- column 1: data
box(ax, 1.5, 31.5, 20, 20, 'Paired data',
    'degraded  $x$   /   clear  $y$\n'
    'A  endoscope, back-lit, 4 levels\n'
    'B, C  camera, ambient light\n'
    'public: Dense-/NH-/O-/I-HAZE',
    'Sec. IV-A, IV-B', C_DATA, E_DATA, 1.2, 8.6, 7.2, align='left')

box(ax, 1.5, 6.0, 20, 21, 'Physical model',
    r'$x=yT+B(1-T)+n$' '\n\n'
    'T nearly constant\n'
    r'$\Rightarrow\ x\approx a\,y+c$' '\n'
    'a GLOBAL AFFINE map',
    'Sec. I, III-A', C_DATA, E_DATA, 1.2, 8.6, 7.2)

# ---------------- column 2: networks and priors
box(ax, 27.5, 31.5, 20.5, 20, 'Restoration net  $f$',
    'NAFNet-Lite  (gated conv.)\n'
    'Restormer-tiny  (channel attn.)\n'
    'ResUNet  (plain residual conv.)\n'
    '14 budgets  $\\times$  7 configs',
    'Sec. IV-C', C_NET, E_NET, 1.2, 8.6, 7.2, align='left')

box(ax, 27.5, 6.0, 20.5, 21, 'Injected physical priors',
    'A  Wiener band prefilter\n'
    'B  geometry (depth / pose)\n'
    'C  medium state scalar\n\n'
    'each arm = one paired contrast',
    'Sec. IV-D', C_NET, E_NET, 1.2, 8.6, 7.2, align='left')

# ---------------- column 3: measurement operators
box(ax, 54.0, 37.5, 25.0, 14.0, '1.  Affine-structure split',
    r'$G_{\rm total}=G_{\rm aff}+G_{\rm struct}-\Delta_o$' '\n'
    'oracle 2-parameter affine\n'
    'vs affine-invariant PSNR',
    'Sec. III-A, V-A to V-C', C_MEAS, E_MEAS, 1.2, 8.4, 7.2)

box(ax, 54.0, 22.0, 25.0, 13.5, '2.  Linear-filter reference',
    'box-blur Laplacian pyramid\n'
    r'per band  $\hat y_k=a_kx_k+c_k$' '\n'
    r'ceiling  ${\rm LFR}_k$  (not a bound on all methods)',
    'Sec. III-B, V-D', C_MEAS, E_MEAS, 1.2, 8.4, 6.6)

box(ax, 54.0, 6.0, 25.0, 14.0, '3.  Paired statistics',
    'scene-clustered bootstrap\n'
    'cross-seed sign consistency\n'
    'PSNR / SSIM / LPIPS / BRISQUE',
    'Sec. IV-F, VI, VII-A', C_MEAS, E_MEAS, 1.2, 8.4, 7.2)

# ---------------- column 4: findings
box(ax, 84.0, 30.5, 14.5, 21.0, 'Findings',
    'S1  affine share > 1/2;\n      falls with NU\n'
    'S2  fine bands at LFR;\n      DC far beyond\n'
    'S3  prior sign set by\n      budget & family',
    '', C_OUT, E_OUT, 1.2, 8.4, 7.0, align='left')

box(ax, 84.0, 6.0, 14.5, 21.0, 'Minimum reporting\nprotocol   (Sec. VIII)',
    '\nreport  $G_{\\rm struct}$\nbeside PSNR\n'
    'report budget curve,\nnot one epoch\n'
    'use measured $\\rho$,\nnot nominal NTU',
    '', C_OUT, E_OUT, 1.2, 8.4, 6.9, align='left')

# ---------------- bottom: cross-session comparability
box(ax, 1.5, 0.4, 97.0, 4.2, '',
    '', '', '#f6f6f6', '#999', 1.0)
ax.text(2.8, 2.5, 'Cross-session (Sec. VII-B):', ha='left', va='center',
        fontsize=8.0, fontweight='bold', zorder=3)
ax.text(27.5, 2.5, 'A / B / C, same protocol:  veiling offset $c$ follows device/illumination;'
                   '  recoverable structure $\\rho$ follows the session',
        ha='left', va='center', fontsize=7.2, color='#222', zorder=3)

# ---------------- arrows
arrow(ax, (21.5, 41.5), (27.5, 41.5))                      # data -> net
arrow(ax, (21.5, 16.5), (27.5, 16.5))                      # model -> priors
arrow(ax, (37.7, 27.0), (37.7, 31.5))                      # priors -> net
arrow(ax, (11.5, 31.5), (11.5, 27.0), ls=(0, (3, 2)))      # data -> model
for yq in (44.5, 28.8, 13.0):
    arrow(ax, (48.0, 41.5), (54.0, yq), rad=0.12)          # net -> measurements
arrow(ax, (79.0, 44.5), (84.0, 42.5), rad=-0.08)
arrow(ax, (79.0, 28.8), (84.0, 40.0), rad=-0.08)
arrow(ax, (79.0, 13.0), (84.0, 34.0), rad=-0.08)
arrow(ax, (91.2, 30.5), (91.2, 27.0))                      # findings -> protocol

# output label
ax.text(50.8, 43.4, r'$o=f(x)$', ha='center', va='bottom', fontsize=7.4, color=GREY, zorder=3)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
figstyle.savefig(fig, OUT)
print('COMPLETE', OUT, flush=True)
