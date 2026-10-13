# -*- coding: utf-8 -*-
"""Paper figure: what a two-parameter photometric map does on public real-haze images.
Rows: selected test images; columns: hazy input / oracle global affine (2 params) / oracle 4x4 tiled
affine (2 params per tile) / 5-fold cross-validated NAFNet-Lite / clear reference.
All panels at the Part-B working resolution (long side 1024); RGB PSNR (dB) in each title.
Restored images: written by `experiments/public_benchmarks.py --save-png` into <RUNS>/<DS>/restored/<idx>.png.
Usage: python figures/fig_public_visual.py DATA_ROOT RUNS_DIR OUT_PREFIX [DS:IDX ...]
       (default DENSE-HAZE:55 NH-HAZE:52; DATA_ROOT/<DS>/_raw holds the hazy/GT pairs)"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'experiments'))
import figstyle
import matplotlib.pyplot as plt
import public_benchmarks as R

ROOT, RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SEL = [s.split(':') for s in (sys.argv[4:] or ['DENSE-HAZE:55', 'NH-HAZE:52'])]
NAME = {'DENSE-HAZE': 'Dense-Haze', 'NH-HAZE': 'NH-HAZE', 'O-HAZE': 'O-HAZE', 'I-HAZE': 'I-HAZE'}
COLS = ['hazy input', 'global affine\n(2 parameters, oracle)', r'$4\times4$ tiled affine' + '\n(32 parameters, oracle)',
        'CV network\n(NAFNet-Lite)', 'clear reference']

rows = []
for ds, k in SEL:
    k = int(k)
    pairs = {i: (h, g) for i, h, g in R.find_pairs(ROOT / ds / '_raw')}
    hp, gp = pairs[k]
    x = R.load(hp, 1024); y = R.load(gp, 1024)
    if y.shape != x.shape:
        y = np.asarray(Image.fromarray((y * 255).astype(np.uint8)).resize(x.shape[1::-1], Image.BICUBIC), np.float32) / 255
    o = np.asarray(Image.open(RUNS / ds / 'restored' / f'{k:02d}.png').convert('RGB'), np.float32) / 255
    if o.shape != x.shape:
        o = np.asarray(Image.fromarray((o * 255).astype(np.uint8)).resize(x.shape[1::-1], Image.BICUBIC), np.float32) / 255
    ga = R.global_affine(x, y); gt = R.tiled_affine(x, y, 4)
    p = [R.psnr(z, y) for z in (x, ga, gt, o)]
    nu = 1 - (p[1] - p[0]) / (p[2] - p[0])
    rows.append((ds, k, [x, ga, gt, o, y], p, nu))
    print(ds, k, ['%.2f' % v for v in p], 'NU=%.2f' % nu, flush=True)

h, w = rows[0][2][0].shape[:2]
fig, axes = plt.subplots(len(rows), 5, figsize=(7.16, len(rows) * 7.16 / 5 * h / w + 0.55), squeeze=False)
for r, (ds, k, ims, p, nu) in enumerate(rows):
    for c in range(5):
        ax = axes[r, c]; ax.imshow(np.clip(ims[c], 0, 1)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(COLS[c], fontsize=7)
        if c < 4:
            txt = f'{p[c]:.2f} dB' + ('' if c == 0 else f' ({p[c] - p[0]:+.2f})')
            ax.text(.03, .04, txt, transform=ax.transAxes, fontsize=6.5, color='white', va='bottom',
                    bbox=dict(facecolor='black', alpha=.55, lw=0, pad=1.2))
    axes[r, 0].set_ylabel(f'{NAME[ds]} #{k}\nNU = {nu:.2f}', fontsize=7)
fig.subplots_adjust(left=.07, right=.995, top=.9 if len(rows) < 3 else .93, bottom=.01, wspace=.03, hspace=.04)
figstyle.savefig(fig, OUT, tight=False)
