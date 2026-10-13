# -*- coding: utf-8 -*-
"""Paper figure: Wiener pre-filter effect at the end of training (background, mean of 4 levels, scene-clustered
95% CI) for every configuration. Usage: python tip/r5_fig_controls.py OUT_PREFIX CONFIG_JSON
CONFIG_JSON: list of {"family","label","csv","epoch"}."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib.pyplot as plt

OUT, CFG = sys.argv[1], json.load(open(sys.argv[2], encoding='utf-8'))
COL = {'gated convolution': '#c0392b', 'channel attention': '#2a78d6', 'plain residual conv.': '#1f8a5b'}
rows = []
for c in CFG:
    if not Path(c['csv']).exists(): continue
    r = pd.read_csv(c['csv']); r.columns = [k.lstrip('﻿') for k in r.columns]
    s = r[(r.arm == 'W:none') & (r.region == 'background') & (r.epoch == c['epoch'])]
    if not len(s): continue
    rows.append(dict(c, m=s['mean'].mean(), lo=s.ci_lo.mean(), hi=s.ci_hi.mean()))
d = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axhspan(-.15, .15, color='#9aa4ad', alpha=.15, lw=0, zorder=0)
ax.axhline(0, color='#444', lw=.7, ls='--')
x = np.arange(len(d))
for i, r in d.iterrows():
    ax.errorbar(i, r.m, yerr=[[r.m - r.lo], [r.hi - r.m]], fmt='o', ms=4, color=COL[r.family], capsize=2.5, lw=1)
ax.set_xticks(x); ax.set_xticklabels(d.label, rotation=55, ha='right', fontsize=6.5)
ax.set_ylabel('Wiener pre-filter effect,\nend of training (dB)', fontsize=7.5)
for fam, c in COL.items():
    if fam in set(d.family): ax.plot([], [], 'o', color=c, label=fam, ms=4)
ax.legend(fontsize=6.3, frameon=False, loc='upper left')
ax.tick_params(labelsize=7)
fig.tight_layout()
figstyle.savefig(fig, OUT)
print(d[['family', 'label', 'm', 'lo', 'hi']].round(3).to_string(index=False))
