# -*- coding: utf-8 -*-
"""Paper figure + table: public benchmarks (runs/public/<DS>/partA.csv, partB.csv).
(a) per dataset: oracle-affine gain vs best published gain on the official test split (native RGB), and
    oracle-affine vs total gain of the 5-fold cross-validated NAFNet-Lite (all pairs).
(b) per image (all four datasets): non-uniformity index NU vs affine share of the CV network's gain;
    binned ratio-of-means and Spearman correlation.
Usage: python tip/r5_fig_public.py [ROOT] [OUT_PREFIX]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / 'runs/public'
OUT = sys.argv[2] if len(sys.argv) > 2 else str(PROJ / 'runs/figs/fig_public')
BEST = {'DENSE-HAZE': (18.34, 'DGFDNet'), 'NH-HAZE': (21.01, 'Fourier-RWKV'),
        'O-HAZE': (25.36, 'ConvIR'), 'I-HAZE': (23.93, 'MSBDN')}
DS = ['DENSE-HAZE', 'O-HAZE', 'I-HAZE', 'NH-HAZE']
NAME = {'DENSE-HAZE': 'Dense-Haze', 'O-HAZE': 'O-HAZE', 'I-HAZE': 'I-HAZE', 'NH-HAZE': 'NH-HAZE'}
COL = {'DENSE-HAZE': '#2a78d6', 'O-HAZE': '#1baf7a', 'I-HAZE': '#eb6834', 'NH-HAZE': '#c0392b'}
rows, pts = [], []
for ds in DS:
    A = pd.read_csv(ROOT / ds / 'partA.csv'); B = pd.read_csv(ROOT / ds / 'partB.csv')
    At = A[A.test]; best, bname = BEST[ds]
    rec = dict(dataset=ds, n_all=len(A), n_test=len(At), best=best, best_method=bname,
               test_psnr_in=At.psnr_in.mean() if len(At) else np.nan,
               test_g_aff=At.g_aff_global.mean() if len(At) else np.nan,
               test_g_aff_ch=At.g_aff_channel.mean() if len(At) else np.nan,
               all_g_aff=A.g_aff_global.mean(), all_g_aff_256=A.g_aff_global_256luma.mean(),
               all_rho=A.rho.mean(), NU=A.NU.mean(),
               cv_g_total=B.gain_total.mean(), cv_g_aff=B.gain_affine.mean(), cv_g_struct=B.gain_struct.mean(),
               cv_psnr_out=B.psnr_out.mean())
    rec['test_g_best'] = best - rec['test_psnr_in']
    rec['test_share'] = rec['test_g_aff'] / rec['test_g_best']
    rec['cv_share'] = rec['cv_g_aff'] / rec['cv_g_total']
    rows.append(rec)
    m = A.merge(B[['idx', 'gain_total', 'gain_affine']], on='idx')
    for _, r in m.iterrows():
        pts.append(dict(dataset=ds, NU=r.NU, g_total=r.gain_total, g_aff=r.gain_affine))
T = pd.DataFrame(rows); P = pd.DataFrame(pts)
T.to_csv(str(OUT) + '_table.csv', index=False)
from scipy.stats import spearmanr
ok = P[P.g_total > 0.5].copy(); ok['share'] = ok.g_aff / ok.g_total
rs, pv = spearmanr(ok.NU, ok.share)
rs_all, pv_all = spearmanr(P.NU, P.g_aff - P.g_total)          # NU vs (affine - total): robust, no ratio
q = pd.qcut(P.NU, 5, duplicates='drop')
binned = P.groupby(q, observed=True).apply(lambda s: pd.Series(dict(NU=s.NU.mean(), share=s.g_aff.mean() / s.g_total.mean(), n=len(s))))

fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]; y = np.arange(len(T)); h = .36
ax.barh(y - h / 2, T.test_g_best, h, color='#d5d8dc', label='best published method')
ax.barh(y + h / 2, T.test_g_aff, h, color=[COL[d] for d in T.dataset], label='oracle 2-parameter affine')
for i, r in T.iterrows():
    ax.text(r.test_g_aff + .15, i + h / 2, f'{100*r.test_share:.0f}%', va='center', fontsize=7.5, weight='bold')
    ax.text(r.test_g_best + .15, i - h / 2, f'{r.best_method}', va='center', fontsize=6.5, color='#555')
ax.set_yticks(y); ax.set_yticklabels([NAME[d] for d in T.dataset], fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel('PSNR gain over the hazy input, official test split (dB)', fontsize=7.5); ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2); ax.set_title('(a)', fontsize=8.5)
ax.set_xlim(0, T.test_g_best.max() * 1.4)
ax = axs[1]
for ds in DS:
    s = ok[ok.dataset == ds]
    ax.scatter(s.NU, s.share.clip(-.5, 2.5), s=9, color=COL[ds], alpha=.75, label=NAME[ds], edgecolor='none')
ax.plot(binned.NU, binned.share, 'k-o', ms=3, lw=1, label='quintile ratio of means')
ax.axhline(1, color='#888', lw=.6, ls=':')
ax.set_xlabel('non-uniformity index NU', fontsize=7.5); ax.set_ylabel('affine share of CV gain', fontsize=7.5)
ax.text(.03, .04, f'Spearman $r$ = {rs:.2f} ($p$ = {pv:.1g}, $n$ = {len(ok)})', transform=ax.transAxes, fontsize=6.8)
from matplotlib.lines import Line2D
hh = [Line2D([], [], ls='', marker='o', ms=4, color=COL[d], label=NAME[d]) for d in DS] + [Line2D([], [], color='k', marker='o', ms=3, lw=1, label='quintile ratio of means')]
ax.legend(handles=hh, fontsize=6, frameon=True, framealpha=.92, edgecolor='none', loc='upper right'); ax.tick_params(labelsize=7); ax.set_title('(b)', fontsize=8.5)
fig.tight_layout(w_pad=1.2)
figstyle.savefig(fig, OUT)
summary = dict(spearman_share=rs, p_share=pv, n_share=len(ok), spearman_affminustotal=rs_all, p2=pv_all,
               binned=binned.reset_index(drop=True).to_dict('records'))
Path(str(OUT) + '_stats.json').write_text(json.dumps(summary, indent=1, default=float))
pd.set_option('display.width', 220)
print(T.round(3).to_string(index=False)); print(json.dumps(summary, indent=1, default=float))
