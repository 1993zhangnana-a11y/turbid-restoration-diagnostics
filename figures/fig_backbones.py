# -*- coding: utf-8 -*-
"""Paper figure: prior effect vs training budget for three architecture families (rows) and three prior
types (columns), each prior shown in the region where it should matter. Shaded bands: scene-clustered 95% CI.
Usage: python tip/r5_fig_backbones.py OUT_PREFIX NAF_SIGNIF RES_SIGNIF [RESUNET_SIGNIF]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib, matplotlib.pyplot as plt

OUT = sys.argv[1]; SRC = sys.argv[2:]
NAMES = ['Gated convolution\n(NAFNet-Lite)', 'Channel attention\n(Restormer-tiny)', 'Plain residual conv.\n(ResUNet)']
RHO = ['3NUT_DAN', '6NUT_DAN', '8NUT_DAN', '11NUT_DAN']; LAB = ['3 NTU', '6 NTU', '8 NTU', '11 NTU']
COL = ['#2a78d6', '#1baf7a', '#eb6834', '#8e44ad']
PAN = [('W:none', 'background', 'Pre-filter (Wiener), background'),
       ('C0:none', 'tool', 'Tool geometry, tool region'),
       ('zero:est', 'full', r'Medium state ($\hat\tau,\hat b$), full frame')]
n = len(SRC)
fig, axes = plt.subplots(n, 3, figsize=(6.2, 1.45 * n + .35), sharex=True, sharey=True, squeeze=False)
for ri, src in enumerate(SRC):
    r = pd.read_csv(src); r.columns = [c.lstrip('﻿') for c in r.columns]
    if 'rho' not in r and 'level' in r: r['rho'] = r['level']
    for ci, (arm, region, title) in enumerate(PAN):
        ax = axes[ri, ci]; s = r[(r.arm == arm) & (r.region == region)]
        for c, rho, lab in zip(COL, RHO, LAB):
            t = s[s.rho == rho].sort_values('epoch')
            if not len(t): continue
            ax.fill_between(t.epoch, t.ci_lo, t.ci_hi, color=c, alpha=.13, lw=0)
            ax.plot(t.epoch, t['mean'], color=c, lw=1.2, marker='o', ms=2, label=lab)
        if len(s):
            last = s[s.epoch == s.epoch.max()]['mean'].mean()
            ax.text(.97, .93, f'end {last:+.2f} dB', transform=ax.transAxes, ha='right', va='top', fontsize=7)
        else:
            ax.text(.5, .5, 'not run', transform=ax.transAxes, ha='center', va='center', fontsize=7, color='#888')
        ax.axhline(0, color='#444', lw=.7, ls='--', zorder=0)
        ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 30, 80])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-1.0, 1.15); ax.tick_params(labelsize=7)
        if ri == 0: ax.set_title(title, fontsize=8)
        if ri == n - 1: ax.set_xlabel('training budget (epochs)', fontsize=7.5)
        if ci == 0: ax.set_ylabel(NAMES[ri] + '\nprior effect (dB)', fontsize=7.5)
h, l = axes[0, 0].get_legend_handles_labels()
axes[0, 2].legend(h, l, fontsize=6.5, frameon=False, ncol=2, loc='lower center')
fig.tight_layout(h_pad=.4, w_pad=.4)
figstyle.savefig(fig, OUT)
