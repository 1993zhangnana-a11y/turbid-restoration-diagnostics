"""Prior effect vs. training budget (Section VII of the paper).

Each arm is trained ONCE and evaluated in situ at a list of epochs, so a 14-point budget curve costs about
one training run. Prior effect = (arm with prior) - (arm without prior) for the same image, seed and epoch.

Arms (`--arms`, comma separated):
  none   no prior (baseline)               geo   tool geometry at the bottleneck (with 30% perturbation)
  state  medium state (tau, b) via FiLM    wiener band-wise Wiener pre-filtered input

Dataset manifest (`--manifest`, CSV): split{train,eval}, degraded, clear, level, scene, geometry, tau, b
  geometry: path to a float32 .npy of shape (5,H,W) [tool_mask, tool_edge, depth_norm, path_norm, confidence]
            (confidence > 0.5 defines the field of view); tau/b may be empty for arms other than `state`.
Wiener coefficients (`--wiener`, JSON): {level: {w:[6], a_dc, c_dc}} estimated on training data only
(see models/wiener.py).

Example:
  python experiments/budget_curve.py --manifest data.csv --wiener wiener.json --family restormer \
      --blocks 1,3,4 --arms none,wiener,geo,state --seeds 0,1,2 --epochs 80 --out runs/restormer
Then: python experiments/signif.py runs/restormer/gain_per_unit.csv runs/restormer/signif.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import build_core, PriorWrapper, count_params, GEO_C
from models.wiener import apply_uint8

LUMA = np.array([.299, .587, .114], np.float32)
EVAL = "2,4,6,8,12,16,20,25,30,40,50,60,70,80"


def load_rgb(p, size):
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))


def load_geo(p, size):
    g = np.load(p).astype(np.float32)
    if g.shape[-1] != size:
        g = F.interpolate(torch.from_numpy(g)[None], size=(size, size), mode="bilinear", align_corners=False)[0].numpy()
    return g


def warp(g, dx=0, dy=0, scale=1):
    th = torch.tensor([[1 / scale, 0, -2 * dx / g.shape[-1] / scale], [0, 1 / scale, -2 * dy / g.shape[-2] / scale]],
                      device=g.device, dtype=g.dtype)[None].repeat(len(g), 1, 1)
    return F.grid_sample(g, F.affine_grid(th, g.shape, align_corners=False), mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def psnr(a, b, m):
    return float(-10 * np.log10(max(float(np.mean((a[m] - b[m]) ** 2)), 1e-12))) if m.sum() >= 16 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--wiener", default="")
    ap.add_argument("--family", default="nafnet", choices=["nafnet", "restormer", "resunet"])
    ap.add_argument("--blocks", default=""); ap.add_argument("--arms", default="none,geo,state,wiener")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--eval-epochs", default=EVAL); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--perturb", type=float, default=0.30, help="geometry perturbation probability (geo arm)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8-sig")))
    tr = [r for r in rows if r["split"] == "train"]; ev = [r for r in rows if r["split"] == "eval"]
    X = np.stack([load_rgb(r["degraded"], a.size) for r in tr]); Y = np.stack([load_rgb(r["clear"], a.size) for r in tr])
    Xe = np.stack([load_rgb(r["degraded"], a.size) for r in ev]); Ye = np.stack([load_rgb(r["clear"], a.size) for r in ev])
    gcache = {}
    G = lambda p: gcache.setdefault(p, load_geo(p, a.size))
    Gtr = np.stack([G(r["geometry"]) for r in tr]); Gev = np.stack([G(r["geometry"]) for r in ev])
    st = lambda rs: np.array([[float(r.get("tau") or 0) / 4, float(r.get("b") or 0) / 32,
                               1.0 if r.get("tau") else 0.0, 0.0] for r in rs], np.float32)
    Ptr, Pev = st(tr), st(ev)
    arms = a.arms.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    evals = sorted({int(v) for v in a.eval_epochs.split(",") if int(v) <= a.epochs} | {a.epochs})
    if "wiener" in arms:
        WC = json.load(open(a.wiener, encoding="utf-8"))
        XW = np.stack([apply_uint8(x, WC[r["level"]]) for x, r in zip(X, tr)])
        XeW = np.stack([apply_uint8(x, WC[r["level"]]) for x, r in zip(Xe, ev)])
    blk = tuple(int(v) for v in a.blocks.split(",")) if a.blocks else None
    gy = (Ye.astype(np.float32) / 255 * LUMA).sum(-1)
    t2 = lambda z: torch.from_numpy(np.ascontiguousarray(z)).permute(0, 3, 1, 2).float().to(dev) / 255
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    out_rows, t0 = [], time.time()
    for arm in arms:
        xa, xea = (XW, XeW) if arm == "wiener" else (X, Xe)
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(712 + seed)
            model = PriorWrapper(build_core(a.family, 3, 32, blk)).to(dev)
            if seed == seeds[0] and arm == arms[0]:
                print(f"{a.family} {blk} params={count_params(model):,}", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
            idx = np.arange(len(xa))
            for ep in range(1, a.epochs + 1):
                model.train(); np.random.shuffle(idx)
                for s in range(0, len(idx), a.batch):
                    ids = idx[s:s + a.batch]
                    gb = torch.from_numpy(Gtr[ids]).to(dev)
                    if arm != "geo":
                        gb = torch.zeros_like(gb)
                    else:
                        for j, ii in enumerate(ids):
                            if rng.random() >= a.perturb: continue
                            k = int(rng.integers(3))
                            if k == 0:
                                dx = int(rng.integers(5, 31)) * int(rng.choice([-1, 1]))
                                dy = int(rng.integers(5, 31)) * int(rng.choice([-1, 1]))
                                gb[j:j + 1] = warp(gb[j:j + 1], dx, dy)
                            elif k == 1:
                                gb[j:j + 1] = warp(gb[j:j + 1], scale=float(rng.uniform(.9, 1.1)))
                            else:
                                others = [q for q in range(len(Gtr)) if tr[q]["geometry"] != tr[ii]["geometry"]]
                                if others: gb[j] = torch.from_numpy(Gtr[int(rng.choice(others))]).to(dev)
                    pr = torch.from_numpy(Ptr[ids] if arm == "state" else np.zeros_like(Ptr[ids])).to(dev)
                    loss = (model(t2(xa[ids]), gb, pr) - t2(Y[ids])).abs().mean()
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
                sch.step()
                if ep in evals:
                    model.eval(); preds = []
                    with torch.inference_mode():
                        for s in range(0, len(xea), 16):
                            gb = torch.from_numpy(Gev[s:s + 16]).to(dev)
                            if arm != "geo": gb = torch.zeros_like(gb)
                            pr = torch.from_numpy(Pev[s:s + 16] if arm == "state" else np.zeros_like(Pev[s:s + 16])).to(dev)
                            o = model(t2(xea[s:s + 16]), gb, pr).cpu().numpy()
                            preds.append((o.transpose(0, 2, 3, 1) * LUMA).sum(-1))
                    P = np.concatenate(preds)
                    for j, r in enumerate(ev):
                        fov = Gev[j][4] > .5; tool = (Gev[j][0] > .5) & fov
                        for nm, reg in (("full", fov), ("tool", tool), ("background", fov & ~tool)):
                            out_rows.append(dict(arm=arm, seed=seed, epoch=ep, level=r["level"], scene=r["scene"],
                                                 uid=j, region=nm, psnr=psnr(P[j], gy[j], reg)))
                    print(f"  {arm} s{seed} ep{ep} ({time.time() - t0:.0f}s)", flush=True)
            del model
            if dev == "cuda": torch.cuda.empty_cache()
    import pandas as pd
    df = pd.DataFrame(out_rows); df.to_csv(od / "curve.csv", index=False)
    key = ["seed", "epoch", "level", "scene", "uid", "region"]
    if "none" in arms:
        b = df[df.arm == "none"].set_index(key)["psnr"]; gains = []
        for arm in arms:
            if arm == "none": continue
            d = (df[df.arm == arm].set_index(key)["psnr"] - b).dropna().reset_index(name="gain"); d["arm"] = arm
            gains.append(d)
        if gains:
            pd.concat(gains).to_csv(od / "gain_per_unit.csv", index=False)
    json.dump(vars(a), open(od / "config.json", "w"), indent=1)


if __name__ == "__main__":
    main()
