"""Paired significance of prior effects along the budget curve.

Input: gain_per_unit.csv with columns arm, seed, epoch, level (or rho), scene (or pos+bg), region, gain.
For every (arm, region, epoch, level): mean, scene-clustered bootstrap 95% CI (4000 resamples, seeds pooled
as replicates), two-sided bootstrap p, per-seed means, their SD, and whether all seeds agree in sign.
Usage: python experiments/signif.py gain_per_unit.csv signif.csv
"""
import sys
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src); d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
if "level" not in d and "rho" in d: d["level"] = d["rho"]
if "scene" not in d: d["scene"] = d["pos"].astype(str) + "_" + d["bg"].astype(str)
rng = np.random.default_rng(20260920); B = 4000
rows = []
for (arm, region, epoch, level), sub in d.groupby(["arm", "region", "epoch", "level"]):
    scenes = sub.scene.unique(); g = {s: sub.loc[sub.scene == s, "gain"].to_numpy() for s in scenes}; n = len(scenes)
    bs = np.array([np.concatenate([g[scenes[i]] for i in rng.integers(0, n, n)]).mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
    ps = sub.groupby("seed")["gain"].mean()
    rows.append(dict(arm=arm, region=region, epoch=epoch, level=level, mean=sub.gain.mean(), ci_lo=lo, ci_hi=hi,
                     p_boot=p, seed_sd=ps.std(ddof=1) if len(ps) > 1 else np.nan,
                     sign_consistent=int(len(set(np.sign(ps.to_numpy()))) == 1), n_units=len(sub), n_scenes=n))
pd.DataFrame(rows).sort_values(["arm", "region", "level", "epoch"]).to_csv(out, index=False)
print("wrote", out)
