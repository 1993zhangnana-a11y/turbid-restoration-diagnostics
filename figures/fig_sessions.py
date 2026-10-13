# -*- coding: utf-8 -*-
"""Paper figure: like-for-like comparison of three acquisition sessions (table_sessions.csv written by the
session script). Sessions are identified by the prefix of the `session` column: A_ (endoscope, back-lit),
B_ and C_ (camera, ambient light).
Usage: python tip/r5_fig_sessions.py TABLE_CSV OUT_PREFIX"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
t = pd.read_csv(SRC); t.columns = [c.lstrip('﻿') for c in t.columns]
t['sess'] = t.session.str[0]
R = ['3NUT_DAN', '6NUT_DAN', '8NUT_DAN', '11NUT_DAN']; LAB = ['3', '6', '8', '11']
S = {'A': ('A  endoscope, back-lit', '#c0392b', 'o'), 'B': ('B  camera, ambient', '#2a78d6', 's'),
     'C': ('C  camera, ambient', '#1baf7a', '^')}
p = t.pivot(index='rho', columns='sess').reindex(R); x = np.arange(4)
fig = plt.figure(figsize=(7.16, 3.95))
gs = fig.add_gridspec(2, 3, hspace=.50, wspace=.48)
PAN = [('rho_global', r'global correlation $\rho$', '(a)'), ('gain_affine', 'oracle affine gain (dB)', '(b)'),
       ('b_fit', r'fitted offset $c$ ($y\approx ax+c$)', '(c)')]
for c, (col, ylab, ttl) in enumerate(PAN):
    ax = fig.add_subplot(gs[0, c])
    for s, (nm, co, mk) in S.items():
        ax.plot(x, p[col][s].to_numpy(), color=co, marker=mk, ms=4, lw=1.2, label=nm, markeredgecolor='white', markeredgewidth=.6,
                zorder=4 if s == 'A' else 3, alpha=1 if s == 'A' else .9)
    if col == 'b_fit': ax.axhline(0, color='#666', lw=.6, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(LAB); ax.set_xlabel('nominal turbidity (NTU)', fontsize=7.5)
    ax.set_ylabel(ylab, fontsize=7.5); ax.set_title(ttl, fontsize=8.5); ax.tick_params(labelsize=7)
    ax.set_axisbelow(True)
fig.axes[2].legend(fontsize=6.3, frameon=False, loc='center right')
ax = fig.add_subplot(gs[1, :])
QS = [('rho_global', r'correlation $\rho$'), ('gain_affine', 'affine gain'), ('b_fit', 'veiling offset $c$'), ('a_fit', 'gain $a$')]
bet, wit = [], []
for col, _ in QS:
    A, B, C = (p[col][k].to_numpy() for k in 'ABC')
    bet.append(np.mean(np.abs(A - (B + C) / 2))); wit.append(np.mean(np.abs(B - C)))
ratio = np.array(bet) / np.array(wit); xx = np.arange(len(QS))
ax.bar(xx, ratio, .5, color=['#8e44ad' if r > 1.8 else '#e59f3a' for r in ratio])
ax.axhline(1, color='#444', lw=.8, ls='--')
for i, (r_, b_, w_) in enumerate(zip(ratio, bet, wit)):
    ax.text(i, r_ + .15, f'{r_:.1f}$\\times$', ha='center', fontsize=8, weight='bold')
ax.set_xticks(xx); ax.set_xticklabels([f'{q[1]}\n(device {b_:.3f} / session {w_:.3f})' for q, b_, w_ in zip(QS, bet, wit)], fontsize=7)
ax.set_ylabel('between-device / between-\nsession dispersion', fontsize=7.5); ax.set_title('(d)', fontsize=8.5)
ax.set_ylim(0, max(ratio) * 1.25); ax.set_xlim(-.6, len(QS) - .4); ax.tick_params(labelsize=7)
ax.set_axisbelow(True)
figstyle.savefig(fig, OUT)
print(pd.DataFrame(dict(q=[q[0] for q in QS], between_device=np.round(bet, 3), between_session=np.round(wit, 3), ratio=np.round(ratio, 2))).to_string(index=False))
