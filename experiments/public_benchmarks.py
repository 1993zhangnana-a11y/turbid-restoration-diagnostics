# -*- coding: utf-8 -*-
"""Public real-haze benchmarks under the protocol used by the literature (Section V-B/C, VI of the paper).

Design: (i) the oracle affine is measured exactly like published results are reported -- RGB PSNR at
native resolution on the official TEST split; (ii) a same-protocol network gives per-image outputs that can
be decomposed and related to the spatial non-uniformity of the degradation.

Part A  (no training) -- per image, native resolution, RGB PSNR (peak 1):
        P_in, oracle global affine (2 params shared by R,G,B), per-channel affine (6 params),
        tiled affine (2 params per tile on a T x T grid), global Pearson rho.
        Non-uniformity index  NU = 1 - G_aff_global / G_aff_tile   (0 = a single global affine is
        as good as a spatially varying one; -> 1 = the degradation is spatially non-uniform).
Part B  (training) -- NAFNet-Lite (identical core to the paper's baseline, no priors) trained by
        K-fold cross-validation on each benchmark at a working resolution, so EVERY image gets an
        out-of-fold restoration; per image we report G_total, G_affine, G_struct (affine-invariant)
        and the band-wise LFR headroom. The official test split is one of the folds.

Outputs  runs/public/<DS>/{partA.csv, partB.csv, bandsB.csv} and runs/public/summary.json
"""
import argparse, json, re, sys, time, importlib
from pathlib import Path
import numpy as np
from PIL import Image

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
# official test indices (1-based, as numbered in the file names)
TEST = {'DENSE-HAZE': range(51, 56), 'NH-HAZE': range(51, 56), 'O-HAZE': range(41, 46), 'I-HAZE': range(31, 36)}
DEG = ('hazy', 'haze', 'gt')  # helper only
W = np.array([.299, .587, .114], np.float32)


def find_pairs(raw):
    """Return sorted list of (idx, hazy_path, gt_path) parsed from the *_hazy / *_GT naming."""
    files = [p for p in Path(raw).rglob('*') if p.suffix.lower() in EXT and '__MACOSX' not in str(p)]
    hz, gt = {}, {}
    for p in files:
        s = p.stem.lower(); m = re.match(r'\D*(\d+)', p.stem)
        if not m: continue
        k = int(m.group(1))
        if 'hazy' in s or 'haze' in s.replace('nh-haze', '').replace('dense', ''):
            hz[k] = p
        elif 'gt' in s or 'clean' in s or 'clear' in s:
            gt[k] = p
    ks = sorted(set(hz) & set(gt))
    return [(k, hz[k], gt[k]) for k in ks]


def load(p, long_side=0):
    im = Image.open(p).convert('RGB')
    if long_side and max(im.size) > long_side:
        s = long_side / max(im.size)
        im = im.resize((max(4, round(im.size[0] * s / 4) * 4), max(4, round(im.size[1] * s / 4) * 4)), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12)))


def aff(x, y):
    xv = x.reshape(-1).astype(np.float64); yv = y.reshape(-1).astype(np.float64)
    v = xv.var()
    if v < 1e-12: return 1.0, float(yv.mean() - xv.mean())
    a = float(((xv - xv.mean()) * (yv - yv.mean())).mean() / v)
    return a, float(yv.mean() - a * xv.mean())


def global_affine(x, y):
    a, c = aff(x, y); return np.clip(a * x + c, 0, 1)


def channel_affine(x, y):
    o = np.empty_like(x)
    for ch in range(3):
        a, c = aff(x[..., ch], y[..., ch]); o[..., ch] = np.clip(a * x[..., ch] + c, 0, 1)
    return o


def tiled_affine(x, y, T):
    H, Wd = x.shape[:2]; o = np.empty_like(x)
    hs = np.linspace(0, H, T + 1).astype(int); ws = np.linspace(0, Wd, T + 1).astype(int)
    for i in range(T):
        for j in range(T):
            sl = (slice(hs[i], hs[i + 1]), slice(ws[j], ws[j + 1]))
            a, c = aff(x[sl], y[sl]); o[sl] = np.clip(a * x[sl] + c, 0, 1)
    return o


def part_a(pairs, ds, tiles):
    rows = []
    for k, hp, gp in pairs:
        x, y = load(hp), load(gp)
        if x.shape != y.shape:
            Image_ = Image.fromarray((y * 255).astype(np.uint8)).resize(x.shape[1::-1], Image.BICUBIC)
            y = np.asarray(Image_, np.float32) / 255
        p_in = psnr(x, y)
        r = dict(dataset=ds, idx=k, test=k in TEST[ds], H=x.shape[0], W=x.shape[1], psnr_in=p_in,
                 rho=float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1]))
        r['g_aff_global'] = psnr(global_affine(x, y), y) - p_in
        r['g_aff_channel'] = psnr(channel_affine(x, y), y) - p_in
        for T in tiles:
            r[f'g_aff_tile{T}'] = psnr(tiled_affine(x, y, T), y) - p_in
        T0 = tiles[0]
        r['NU'] = 1 - r['g_aff_global'] / r[f'g_aff_tile{T0}'] if r[f'g_aff_tile{T0}'] > 1e-6 else np.nan
        # 256x256 luma (legacy protocol) for continuity with the earlier table
        xs = (np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
                         np.float32) / 255 * W).sum(-1)
        ys = (np.asarray(Image.fromarray((y * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR),
                         np.float32) / 255 * W).sum(-1)
        a_, c_ = aff(xs, ys)
        r['g_aff_global_256luma'] = psnr(np.clip(a_ * xs + c_, 0, 1), ys) - psnr(xs, ys)
        rows.append(r)
        print(f'  A {ds} #{k:02d} {x.shape[1]}x{x.shape[0]} in={p_in:.2f} affG={r["g_aff_global"]:.2f} '
              f'affT={r[f"g_aff_tile{T0}"]:.2f} NU={r["NU"]:.2f}', flush=True)
    return rows


def part_b(pairs, ds, a, gd):
    import torch, torch.nn as nn
    from models import build_core, GEO_C
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = [load(hp, a.long_side) for _, hp, _ in pairs]
    Y = []
    for (_, _, gp), x in zip(pairs, X):
        y = load(gp, a.long_side)
        if y.shape != x.shape:
            y = np.asarray(Image.fromarray((y * 255).astype(np.uint8)).resize(x.shape[1::-1], Image.BICUBIC),
                           np.float32) / 255
        Y.append(y)
    n = len(pairs); folds = np.array_split(np.arange(n), a.folds)
    pred = [None] * n
    for f, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        torch.manual_seed(a.seed + f); rng = np.random.default_rng(a.seed + f)
        model = build_core('nafnet', 3, 32, (2, 2, 4)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
        g0 = torch.zeros(a.batch, GEO_C, 8, 8, device=dev); p0 = torch.zeros(a.batch, 4, device=dev)
        t0 = time.time(); ls = []
        for it in range(1, a.iters + 1):
            xb, yb = [], []
            for _ in range(a.batch):
                i = int(rng.choice(tr)); x, y = X[i], Y[i]
                hh, ww = x.shape[:2]; cs = min(a.crop, hh, ww) // 4 * 4
                r0 = int(rng.integers(0, hh - cs + 1)); c0 = int(rng.integers(0, ww - cs + 1))
                xc, yc = x[r0:r0 + cs, c0:c0 + cs], y[r0:r0 + cs, c0:c0 + cs]
                if rng.random() < .5: xc, yc = xc[:, ::-1], yc[:, ::-1]
                if rng.random() < .5: xc, yc = xc[::-1], yc[::-1]
                xb.append(xc.transpose(2, 0, 1).copy()); yb.append(yc.transpose(2, 0, 1).copy())
            xb = torch.from_numpy(np.stack(xb)).to(dev); yb = torch.from_numpy(np.stack(yb)).to(dev)
            model.train()
            loss = (model(xb, g0, p0) - yb).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step(); sch.step(); ls.append(float(loss.detach()))
            if it % 1000 == 0:
                print(f'  B {ds} fold{f} it{it} loss={np.mean(ls[-1000:]):.4f} ({time.time()-t0:.0f}s)', flush=True)
        model.eval()
        with torch.inference_mode():
            for i in te:
                x = X[i]; hh, ww = x.shape[:2]
                ph, pw = (-hh) % 4, (-ww) % 4
                xi = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode='reflect')
                o = model(torch.from_numpy(xi.transpose(2, 0, 1)[None].copy()).to(dev),
                          torch.zeros(1, GEO_C, 8, 8, device=dev), torch.zeros(1, 4, device=dev))
                pred[i] = np.clip(o[0].cpu().numpy().transpose(1, 2, 0)[:hh, :ww], 0, 1)
        del model; torch.cuda.empty_cache()
    rows, brows = [], []
    for i, (k, _, _) in enumerate(pairs):
        x, y, o = X[i], Y[i], pred[i]
        p_in = psnr(x, y); p_out = psnr(o, y)
        p_aff_in = psnr(global_affine(x, y), y); p_aff_out = psnr(global_affine(o, y), y)
        rows.append(dict(dataset=ds, idx=k, test=k in TEST[ds], psnr_in=p_in, psnr_out=p_out,
                         gain_total=p_out - p_in, gain_affine=p_aff_in - p_in,
                         gain_struct=p_aff_out - p_aff_in))
        xl, yl, ol = (x * W).sum(-1), (y * W).sum(-1), (o * W).sum(-1)
        m = np.ones_like(yl, bool)
        bg_ = gd.band_gain(xl, ol, yl, m); bc = gd.band_ceiling(xl, yl, m)
        for b, nm in enumerate(gd.BAND_NAMES):
            brows.append(dict(dataset=ds, idx=k, band=nm, lfr_db=bc[b]['ceiling_db'], gain_db=bg_[b]))
        if a.save_png and k in TEST[ds]:
            od = PROJ / a.out / ds / 'restored'; od.mkdir(parents=True, exist_ok=True)
            Image.fromarray((o * 255 + .5).astype(np.uint8)).save(od / f'{k:02d}.png')
    return rows, brows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder with <DATASET>/_raw/ unpacked NTIRE archives')
    ap.add_argument('--datasets', default='DENSE-HAZE,NH-HAZE,O-HAZE,I-HAZE')
    ap.add_argument('--out', default='runs/public')
    ap.add_argument('--tiles', default='4,8')
    ap.add_argument('--skip-a', action='store_true'); ap.add_argument('--skip-b', action='store_true')
    ap.add_argument('--long-side', type=int, default=1024, help='Part B working resolution (long side, px)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-png', action='store_true')
    a = ap.parse_args()
    import pandas as pd
    from tdiag import lfr as gd
    tiles = [int(t) for t in a.tiles.split(',')]
    summ = {}
    for ds in a.datasets.split(','):
        raw = Path(a.root) / ds / '_raw'
        pairs = find_pairs(raw)
        if a.limit: pairs = pairs[:a.limit]
        print(f'[{ds}] {len(pairs)} pairs; indices {pairs[0][0]}..{pairs[-1][0]}; '
              f'test present: {sorted(k for k, _, _ in pairs if k in TEST[ds])}', flush=True)
        od = PROJ / a.out / ds; od.mkdir(parents=True, exist_ok=True)
        s = dict(n=len(pairs), idx=[k for k, _, _ in pairs],
                 names=[(Path(h).name, Path(g).name) for _, h, g in pairs][:3])
        if not a.skip_a:
            A = pd.DataFrame(part_a(pairs, ds, tiles)); A.to_csv(od / 'partA.csv', index=False)
            s['A_all'] = A.drop(columns=['dataset', 'test']).mean(numeric_only=True).to_dict()
            if A.test.any(): s['A_test'] = A[A.test].drop(columns=['dataset', 'test']).mean(numeric_only=True).to_dict()
        if not a.skip_b:
            rows, brows = part_b(pairs, ds, a, gd)
            B = pd.DataFrame(rows); B.to_csv(od / 'partB.csv', index=False)
            pd.DataFrame(brows).to_csv(od / 'bandsB.csv', index=False)
            for nm, sub in (('B_all', B), ('B_test', B[B.test])):
                if len(sub):
                    d = sub.drop(columns=['dataset', 'test']).mean(numeric_only=True).to_dict()
                    d['share'] = d['gain_affine'] / d['gain_total'] if abs(d['gain_total']) > 1e-6 else None
                    s[nm] = d
        summ[ds] = s
        print(json.dumps({ds: {k: v for k, v in s.items() if k != 'idx'}}, indent=1, default=float), flush=True)
    (PROJ / a.out).mkdir(parents=True, exist_ok=True)
    (PROJ / a.out / 'summary.json').write_text(json.dumps(summ, indent=1, default=float), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
