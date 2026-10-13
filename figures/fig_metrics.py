# -*- coding: utf-8 -*-
"""Paper figure: metric dependence on the full held-out set (runs/heldout/table_metrics.csv).
(a) share of improvement delivered by a two-parameter oracle affine, per metric;
(b) PSNR gain vs LPIPS improvement with scene-clustered 95% CIs; (c) seed averaging: PSNR vs BRISQUE.
Usage: python tip/r5_fig_metrics.py [SRC_DIR] [OUT_PREFIX]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / 'runs/heldout'
OUT = sys.argv[2] if len(sys.argv) > 2 else str(PROJ / 'runs/figs/fig_metrics')
T = pd.read_csv(SRC / 'table_metrics.csv'); T.columns = [c.lstrip('﻿') for c in T.columns]
R = ['3NUT_DAN', '6NUT_DAN', '8NUT_DAN', '11NUT_DAN']; LAB = ['3', '6', '8', '11']
g = lambda m, r, c: float(T[(T.method == m) & (T.rho == r)].iloc[0][c])
x = np.arange(4)
fig = plt.figure(figsize=(7.16, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=.95)

ax = fig.add_subplot(gs[0]); w = .26
sh = {mt: [100 * g('affine_oracle', r, mt + '_delta') / g('final_avg', r, mt + '_delta') for r in R]
      for mt in ('psnr', 'ssim', 'lpips')}
cols = {'psnr': '#c0392b', 'ssim': '#2a78d6', 'lpips': '#1baf7a'}
for k, (mt, nm) in enumerate((('psnr', 'PSNR'), ('ssim', 'SSIM'), ('lpips', 'LPIPS'))):
    ax.bar(x + (k - 1) * w, sh[mt], w, color=cols[mt], label=nm)
for i, v in enumerate(sh['psnr']):
    ax.text(i - w, v + 2, f'{v:.0f}', ha='center', fontsize=7, color=cols['psnr'])
ax.axhline(0, color='#444', lw=.8)
ax.set_xticks(x); ax.set_xticklabels(LAB); ax.set_xlabel('nominal turbidity (NTU)'); ax.set_ylim(-40, 95)
ax.set_ylabel('oracle-affine share of\nimprovement (%)')
ax.set_title('(a)', fontsize=9)
ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=1)
ax.set_axisbelow(True)

ax = fig.add_subplot(gs[1])
p = np.array([g('final_avg', r, 'psnr_delta') for r in R])
plo = np.array([g('final_avg', r, 'psnr_lo') for r in R]); phi = np.array([g('final_avg', r, 'psnr_hi') for r in R])
l = -np.array([g('final_avg', r, 'lpips_delta') for r in R])
llo = -np.array([g('final_avg', r, 'lpips_hi') for r in R]); lhi = -np.array([g('final_avg', r, 'lpips_lo') for r in R])
ax.errorbar(x, p, yerr=[p - plo, phi - p], color='#c0392b', marker='o', ms=4, lw=1.4, capsize=2.5)
ax2 = ax.twinx()
ax2.errorbar(x, l, yerr=[l - llo, lhi - l], color='#1baf7a', marker='s', ms=4, lw=1.4, capsize=2.5)
ax2.axhline(0, color='#1baf7a', lw=.7, ls=':')
ax.set_ylim(0, 6.5); lim = max(abs(l).max() * 1.25, .05); ax2.set_ylim(-lim, lim)
ax.set_xticks(x); ax.set_xticklabels(LAB); ax.set_xlabel('nominal turbidity (NTU)')
ax.set_ylabel('ΔPSNR (dB)', color='#c0392b'); ax.tick_params(axis='y', colors='#c0392b')
ax2.set_ylabel('LPIPS improvement', color='#1baf7a'); ax2.tick_params(axis='y', colors='#1baf7a')
ax.set_title('(b)', fontsize=9); ax.set_axisbelow(True)

ax = fig.add_subplot(gs[2])
for m in ('final_s0', 'final_s1', 'final_s2', 'final_avg'):
    xs = [g(m, r, 'brisque') for r in R]; ys = [g(m, r, 'psnr_delta') for r in R]
    ax.scatter(xs, ys, c='#c0392b' if m == 'final_avg' else '#9aa4ad', marker='D' if m == 'final_avg' else 'o',
               s=26 if m == 'final_avg' else 16, zorder=3, edgecolor='white', lw=.6,
               label={'final_s0': 'single seeds', 'final_avg': '3-seed average'}.get(m))
for r in R:
    xs = np.mean([g(m, r, 'brisque') for m in ('final_s0', 'final_s1', 'final_s2')])
    ys = np.mean([g(m, r, 'psnr_delta') for m in ('final_s0', 'final_s1', 'final_s2')])
    ax.annotate('', xy=(g('final_avg', r, 'brisque'), g('final_avg', r, 'psnr_delta')), xytext=(xs, ys),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=.8, alpha=.6))
ax.set_xlabel('BRISQUE (lower = more natural)'); ax.set_ylabel('ΔPSNR (dB)')
ax.set_title('(c)', fontsize=9); ax.legend(fontsize=7, frameon=False, loc='upper left')
ax.set_axisbelow(True)
figstyle.savefig(fig, OUT)
print({k: np.round(v, 1).tolist() for k, v in sh.items()})
