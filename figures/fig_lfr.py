# -*- coding: utf-8 -*-
"""Paper figure: headroom to the per-band linear filter reference (LFR), finalised baseline,
full held-out set (runs/heldout/table_lfr.csv).
Usage: python tip/r5_fig_lfr.py [--src CSV] [--out PREFIX] [--method final_avg]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument('--src', default=str(PROJ / 'runs/heldout/table_lfr.csv'))
ap.add_argument('--out', default=str(PROJ / 'runs/figs/fig_lfr_headroom'))
ap.add_argument('--method', default='final_avg')
a = ap.parse_args()
t = pd.read_csv(a.src); t.columns = [c.lstrip('﻿') for c in t.columns]
t = t[t.method == a.method]
B = ['<2px', '2-4px', '4-8px', '8-16px', '16-32px', '>32px(DC)']
BL = ['< 2 px', '2–4 px', '4–8 px', '8–16 px', '16–32 px', '> 32 px (DC)']
R = ['3NUT_DAN', '6NUT_DAN', '8NUT_DAN', '11NUT_DAN']; L = ['3 NTU', '6 NTU', '8 NTU', '11 NTU']
M = t.pivot_table(index='band', columns='rho', values='headroom_db').reindex(B)[R].to_numpy(float)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
v = np.abs(M).max()
im = ax.imshow(M, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
for i in range(len(B)):
    for j in range(4):
        ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(M[i, j]) > .55 * v else '#222')
ax.set_xticks(range(4)); ax.set_xticklabels(L)
ax.set_yticks(range(len(B))); ax.set_yticklabels(BL)
ax.set_xlabel('nominal turbidity'); ax.set_ylabel('spatial band')
cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03)
cb.set_label('LFR$_k$ − measured band gain (dB)')
figstyle.savefig(fig, a.out)
print(pd.DataFrame(M, index=B, columns=L).round(2).to_string())
