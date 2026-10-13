# -*- coding: utf-8 -*-
"""Like-for-like comparison of independent acquisition sessions (Section IX of the paper).

The three sessions have different folder layouts but are measured with exactly the same protocol: the same
decomposition code, the same size (256 x 256 luminance) and the same mask rule (clear luminance > 0.02).
  A  endoscope, back-lit   position_N/data_train/<level>/<split>/<bg>/*.png  <->  data_label/<label>/<bg>/
  B  camera, ambient light position_N/data_train/<level>/<split>/x.bmp      <->  data_label/<label>/x.bmp
  C  camera, ambient light data_train/<level>/<split>/x.png                 <->  data_label/<label>/x.png
Usage: python experiments/sessions.py --root-a A --root-b B --root-c C --out runs/sessions
"""
import argparse, sys, json
from pathlib import Path
import numpy as np

RHOS = ['3NUT_DAN', '6NUT_DAN', '8NUT_DAN', '11NUT_DAN']
LABEL_OF = {'train_input': 'train_label', 'test_input': 'test_label'}
IMG = ('.png', '.bmp', '.jpg', '.jpeg', '.tif')


def _files(d):
    p = Path(d)
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG]) if p.is_dir() else []


def pairs_A(root, per_unit):
    """endoscope: one clear frame per (pose, background)"""
    out = []
    for pdir in sorted(Path(root).glob('position_*')):
        for rho in RHOS:
            rd = pdir / 'data_train' / rho
            if not rd.is_dir(): continue
            for sp, lab in LABEL_OF.items():
                sd = rd / sp
                if not sd.is_dir(): continue
                for bg in sorted([b for b in sd.iterdir() if b.is_dir()],
                                 key=lambda s: int(s.name) if s.name.isdigit() else 0):
                    cl = _files(pdir / 'data_label' / lab / bg.name)
                    dg = _files(bg)
                    if not cl or not dg: continue
                    for f in dg[::max(1, len(dg) // per_unit)][:per_unit]:
                        out.append((rho, str(f), str(cl[0])))
    return out


def pairs_B(root, per_unit):
    """camera session B: same file name pairing"""
    out = []
    for pdir in sorted(Path(root).glob('position_*')):
        for rho in RHOS:
            for sp, lab in LABEL_OF.items():
                dg = _files(pdir / 'data_train' / rho / sp)
                cd = pdir / 'data_label' / lab
                for f in dg[::max(1, len(dg) // (per_unit * 20))][:per_unit * 20]:
                    c = cd / f.name
                    if c.exists(): out.append((rho, str(f), str(c)))
    return out


def pairs_C(root, per_unit):
    """camera session C: same file name pairing"""
    out = []
    for rho in RHOS:
        for sp, lab in LABEL_OF.items():
            dg = _files(Path(root) / 'data_train' / rho / sp)
            cd = Path(root) / 'data_label' / lab
            for f in dg:
                c = cd / f.name
                if c.exists(): out.append((rho, str(f), str(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--per-unit', type=int, default=2, help='frames per unit in session A')
    ap.add_argument('--mask-thr', type=float, default=0.02)
    ap.add_argument('--root-a', required=True)
    ap.add_argument('--root-b', required=True)
    ap.add_argument('--root-c', required=True)
    a = ap.parse_args()
    project = Path(a.project); sys.path.insert(0, str(project))
    from tdiag import decomp as _d, lfr as _l
    from tdiag.io import load as _load, luma as _luma
    gd = type('gd', (), dict(psnr=staticmethod(_d.psnr), affine_fit=staticmethod(_d.affine_fit),
               band_ceiling=staticmethod(_l.band_ceiling), BAND_NAMES=_l.BAND_NAMES,
               luma=staticmethod(lambda p, size: _luma(_load(p, size)))))
    import time as _t

    def luma(path, size, tries=8):
        """retry transient file-system errors before giving up"""
        for k in range(tries):
            try:
                return gd.luma(path, size)
            except (PermissionError, OSError) as e:
                if k == tries - 1:
                    print(f'  [skip] {path}: {e}', flush=True); return None
                _t.sleep(0.25 * (k + 1))

    SESS = [
        ('A_endoscope', a.root_a, pairs_A),
        ('B_camera', a.root_b, pairs_B),
        ('C_camera', a.root_c, pairs_C),
    ]
    rows = []
    for name, root, fn in SESS:
        if not Path(root).is_dir():
            print(f'[skip] {name}: not found {root}', flush=True); continue
        prs = fn(root, a.per_unit)
        print(f'{name}: {len(prs)} pairs', flush=True)
        for rho, dp, cp in prs:
            x = luma(dp, a.size); y = luma(cp, a.size)
            if x is None or y is None: continue
            m = y > a.mask_thr
            if m.sum() < 256: m = np.ones_like(m, bool)
            xa, ya = x[m], y[m]
            r = float(np.corrcoef(xa, ya)[0, 1]) if xa.std() > 1e-8 and ya.std() > 1e-8 else np.nan
            aa, cc = gd.affine_fit(x, y, m)
            p_in = gd.psnr(x, y, m)
            p_af = gd.psnr(np.clip(aa * x + cc, 0, 1), y, m)
            bc = gd.band_ceiling(x, y, m)
            rec = dict(session=name, rho=rho, frame=Path(dp).name,
                       psnr_in=p_in, psnr_affine=p_af, gain_affine=p_af - p_in,
                       rho_global=r, a_fit=aa, b_fit=cc)
            for k, nm in enumerate(gd.BAND_NAMES):
                rec['ceil_' + nm] = bc[k]['ceiling_db']; rec['rho_' + nm] = bc[k]['rho']
            rows.append(rec)

    import pandas as pd
    df = pd.DataFrame(rows)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    df.to_csv(od / 'per_image.csv', index=False, encoding='utf-8-sig')
    cols = ['psnr_in', 'psnr_affine', 'gain_affine', 'rho_global', 'a_fit', 'b_fit'] + \
           ['ceil_' + n for n in gd.BAND_NAMES]
    t = df.groupby(['session', 'rho'])[cols].mean().round(3)
    t['n'] = df.groupby(['session', 'rho']).size()
    t = t.reindex(pd.MultiIndex.from_product([sorted(df.session.unique()), RHOS], names=['session', 'rho']))
    t.to_csv(od / 'table_sessions.csv', encoding='utf-8-sig')
    print('\n=== sessions, same protocol (256, mask = clear luma > 0.02) ===')
    print(t[['n', 'psnr_in', 'psnr_affine', 'gain_affine', 'rho_global', 'a_fit', 'b_fit']].to_string())
    print('\n=== spread of rho across sessions for the same nominal label ===')
    p = t['rho_global'].unstack('session')
    p['range'] = (p.max(1) - p.min(1)).round(3)
    print(p.round(3).to_string())
    json.dump(dict(size=a.size, mask_thr=a.mask_thr, per_unit=a.per_unit,
                   n=int(len(df))), open(od / 'config.json', 'w'), indent=1)
    print(f'\nwrote {od}')


if __name__ == '__main__':
    main()
